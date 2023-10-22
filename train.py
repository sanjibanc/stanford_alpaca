#    Copyright 2023 Rohan Taori, Ishaan Gulrajani, Tianyi Zhang, Yann Dubois, Xuechen Li
#
#    Licensed under the Apache License, Version 2.0 (the "License");
#    you may not use this file except in compliance with the License.
#    You may obtain a copy of the License at
#
#        http://www.apache.org/licenses/LICENSE-2.0
#
#    Unless required by applicable law or agreed to in writing, software
#    distributed under the License is distributed on an "AS IS" BASIS,
#    WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#    See the License for the specific language governing permissions and
#    limitations under the License.

import os
import copy
import json
import logging
from dataclasses import dataclass, field
from typing import Optional, Dict, Sequence
import pathlib

import torch
import transformers
from torch.utils.data import Dataset
from transformers import Trainer as HFTrainer
from transformers.utils import logging as hf_logging
logger = hf_logging.get_logger(__name__)
from multiprocessing import Pool
from tqdm.auto import tqdm
from functools import partial

from transformers.trainer import (
    TRAINING_ARGS_NAME,
    WEIGHTS_NAME,
    unwrap_model,
)

import utils

IGNORE_INDEX = -100
DEFAULT_PAD_TOKEN = "[PAD]"
DEFAULT_EOS_TOKEN = "</s>"
DEFAULT_BOS_TOKEN = "</s>"
DEFAULT_UNK_TOKEN = "</s>"
PROMPT_DICT = {
    "prompt_input": (
        "Below is an instruction that describes a task, paired with an input that provides further context. "
        "Write a response that appropriately completes the request.\n\n"
        "### Instruction:\n{instruction}\n\n### Input:\n{input}\n\n### Response:"
    ),
    "prompt_no_input": (
        "Below is an instruction that describes a task. "
        "Write a response that appropriately completes the request.\n\n"
        "### Instruction:\n{instruction}\n\n### Response:"
    ),
}
PROMPT_DICTS = [
    {
        "prompt_input": (
            "Below is an instruction that describes a task, paired with an input that provides further context. "
            "Write a response that appropriately completes the request.\n\n"
            "### Instruction:\n{instruction}\n\n### Input:\n{input}\n\n### Response:"
        ),
        "prompt_no_input": (
            "Below is an instruction that describes a task. "
            "Write a response that appropriately completes the request.\n\n"
            "### Instruction:\n{instruction}\n\n### Response:"
        ),
    },
    {
        "prompt_input": "Below is an instruction that describes a task. Write a response that appropriately completes the request.\n\n### Instruction:\n{instruction}\n\nInput:\n{input}\n\n### Response:\n",
        "prompt_no_input": "Below is an instruction that describes a task. Write a response that appropriately completes the request.\n\n### Instruction:\n{instruction}\n\n### Response:\n",
    }, # dolly style
    {
        "prompt_input": "{instruction}\n\nInput:\n{input}\n\n",
        "prompt_no_input": "{instruction}\n",
    }, # plain
    
]


## overwrite the trainer to support LoRA saving
class Trainer(HFTrainer):
    def _save(self, output_dir: Optional[str] = None, state_dict=None):
        # If we are executing this function, we are the process zero, so we don't check for that.
        output_dir = output_dir if output_dir is not None else self.args.output_dir
        os.makedirs(output_dir, exist_ok=True)
        logger.info(f"Saving model checkpoint to {output_dir}")
        # Save a trained model and configuration using `save_pretrained()`.
        # They can then be reloaded using `from_pretrained()`
        if not hasattr(self.model, "save_pretrained"):
            if hasattr(unwrap_model(self.model), "save_pretrained"):
                if state_dict is None:
                    state_dict = self.model.state_dict()
                unwrap_model(self.model).save_pretrained(output_dir, state_dict=state_dict)
            elif hasattr(self.model, "save_pretrained"):
                self.model.save_pretrained(output_dir, state_dict=state_dict)
            else:
                logger.info("Trainer.model doesn't have a `save_pretrained` method, only saving its state dict.")
                if state_dict is None:
                    state_dict = self.model.state_dict()
                torch.save(state_dict, os.path.join(output_dir, WEIGHTS_NAME))
        else:
            self.model.save_pretrained(output_dir, state_dict=state_dict)
        if self.tokenizer is not None:
            self.tokenizer.save_pretrained(output_dir)

        # Good practice: save your training arguments together with the trained model
        torch.save(self.args, os.path.join(output_dir, TRAINING_ARGS_NAME))

@dataclass
class ModelArguments:
    model_name_or_path: Optional[str] = field(default="facebook/opt-125m")
    use_lora: bool = field(default=False)
    lora_r: int = field(
        default=16,
        metadata={"help": "LoRA r"},
    )
    lora_alpha: int = field(
        default=32,
        metadata={"help": "LoRA alpha"},
    )
    lora_dropout: int = field(
        default=0.05,
        metadata={"help": "LoRA dropout"},
    )
    use_flash_attention: bool = field(
        default=False,
        metadata={"help": "Use flash attention"},
    )
    flash_attention_version: str = field(
        default="torch2",
        metadata={"help": "flash attention version. Should be either 'torch2', 'triton', or 'flash'"},
    )
    use_qlora: bool = field(default=False)


@dataclass
class DataArguments:
    data_path: str = field(default=None, metadata={"help": "Path to the training data."})
    eval_data_path: str = field(default=None, metadata={"help": "Path to the training data."})


@dataclass
class TrainingArguments(transformers.TrainingArguments):
    cache_dir: Optional[str] = field(default=None)
    optim: str = field(default="adamw_torch")
    model_max_length: int = field(
        default=512,
        metadata={"help": "Maximum sequence length. Sequences will be right padded (and possibly truncated)."},
    )
    fast_tokenizer: bool = field(default=False)


def safe_save_model_for_hf_trainer(trainer: transformers.Trainer, output_dir: str):
    """Collects the state dict and dump to disk."""
    # state_dict = trainer.model.state_dict()
    # if trainer.args.should_save:
    #     cpu_state_dict = {key: value.cpu() for key, value in state_dict.items()}
    #     del state_dict
    trainer.save_model(output_dir)  # noqa


def smart_tokenizer_and_embedding_resize(
    special_tokens_dict: Dict,
    tokenizer: transformers.PreTrainedTokenizer,
    model: transformers.PreTrainedModel,
):
    """Resize tokenizer and embedding.

    Note: This is the unoptimized version that may make your embedding size not be divisible by 64.
    """
    num_new_tokens = tokenizer.add_special_tokens(special_tokens_dict)
    model.resize_token_embeddings(len(tokenizer))

    if num_new_tokens > 0:
        input_embeddings = model.get_input_embeddings().weight.data
        output_embeddings = model.get_output_embeddings().weight.data

        input_embeddings_avg = input_embeddings[:-num_new_tokens].mean(dim=0, keepdim=True)
        output_embeddings_avg = output_embeddings[:-num_new_tokens].mean(dim=0, keepdim=True)

        input_embeddings[-num_new_tokens:] = input_embeddings_avg
        output_embeddings[-num_new_tokens:] = output_embeddings_avg


def _tokenize_fn(strings: Sequence[str], tokenizer: transformers.PreTrainedTokenizer) -> Dict:
    """Tokenize a list of strings."""
    tokenized_list = [
        tokenizer(
            text,
            return_tensors="pt",
            padding="longest",
            max_length=tokenizer.model_max_length,
            truncation=True,
        )
        # for text in tqdm(strings, desc="tokenizing", mininterval=0.3)
        for text in strings
    ]
    input_ids = labels = [tokenized.input_ids[0] for tokenized in tokenized_list]
    input_ids_lens = labels_lens = [
        tokenized.input_ids.ne(tokenizer.pad_token_id).sum().item() for tokenized in tokenized_list
    ]
    return dict(
        input_ids=input_ids,
        labels=labels,
        input_ids_lens=input_ids_lens,
        labels_lens=labels_lens,
    )


def preprocess(
    sources: Sequence[str],
    targets: Sequence[str],
    tokenizer: transformers.PreTrainedTokenizer,
) -> Dict:
    """Preprocess the data by tokenizing."""
    examples = [s + t for s, t in zip(sources, targets)]
    examples_tokenized, sources_tokenized = [_tokenize_fn(strings, tokenizer) for strings in (examples, sources)]
    input_ids = examples_tokenized["input_ids"]
    labels = copy.deepcopy(input_ids)
    for label, source_len in zip(labels, sources_tokenized["input_ids_lens"]):
        label[:source_len] = IGNORE_INDEX
    return dict(input_ids=input_ids, labels=labels)


class SupervisedDataset(Dataset):
    """Dataset for supervised fine-tuning."""

    def __init__(self, data_path: str, tokenizer: transformers.PreTrainedTokenizer):
        super(SupervisedDataset, self).__init__()
        logging.warning("Loading data...")
        if data_path.endswith('.jsonl'):
            with open(data_path) as f:
                list_data_dict = [json.loads(line) for line in f]
        else:
            list_data_dict = utils.jload(data_path)

        logging.warning("Formatting inputs...")
        prompt_input, prompt_no_input = PROMPT_DICT["prompt_input"], PROMPT_DICT["prompt_no_input"]
        sources = [
            prompt_input.format_map(example) if example.get("input", "") != "" else prompt_no_input.format_map(example)
            for example in list_data_dict
        ]
        targets = [f"{example['output']}{tokenizer.eos_token}" for example in list_data_dict]

        # logging.warning("Tokenizing inputs... This may take some time...")
        # data_dict = preprocess(sources, targets, tokenizer)
        self.sources = sources
        self.targets = targets
        self.tokenizer = tokenizer

        # self.input_ids = data_dict["input_ids"]
        # self.labels = data_dict["labels"]
        self.input_ids = {}
        self.labels = {}

    def __len__(self):
        return len(self.sources)

    def __getitem__(self, i) -> Dict[str, torch.Tensor]:
        if i not in self.input_ids:
            data_dict = preprocess([self.sources[i]], [self.targets[i]], self.tokenizer)
            self.input_ids[i] = data_dict["input_ids"][0]
            self.labels[i] = data_dict["labels"][0]
            
        return dict(input_ids=self.input_ids[i], labels=self.labels[i])


@dataclass
class DataCollatorForSupervisedDataset(object):
    """Collate examples for supervised fine-tuning."""

    tokenizer: transformers.PreTrainedTokenizer

    def __call__(self, instances: Sequence[Dict]) -> Dict[str, torch.Tensor]:
        input_ids, labels = tuple([instance[key] for instance in instances] for key in ("input_ids", "labels"))
        input_ids = torch.nn.utils.rnn.pad_sequence(
            input_ids, batch_first=True, padding_value=self.tokenizer.pad_token_id
        )
        labels = torch.nn.utils.rnn.pad_sequence(labels, batch_first=True, padding_value=IGNORE_INDEX)
        return dict(
            input_ids=input_ids,
            labels=labels,
            attention_mask=input_ids.ne(self.tokenizer.pad_token_id),
        )


def make_supervised_data_module(tokenizer: transformers.PreTrainedTokenizer, data_args) -> Dict:
    """Make dataset and collator for supervised fine-tuning."""
    train_dataset = SupervisedDataset(tokenizer=tokenizer, data_path=data_args.data_path)
    data_collator = DataCollatorForSupervisedDataset(tokenizer=tokenizer)
    if data_args.eval_data_path is not None:
        eval_dataset = SupervisedDataset(tokenizer=tokenizer, data_path=data_args.eval_data_path)
    else:
        eval_dataset = None
    return dict(train_dataset=train_dataset, eval_dataset=eval_dataset, data_collator=data_collator)


def train():
    parser = transformers.HfArgumentParser((ModelArguments, DataArguments, TrainingArguments))
    model_args, data_args, training_args = parser.parse_args_into_dataclasses()

    tokenizer = transformers.AutoTokenizer.from_pretrained(
        model_args.model_name_or_path,
        cache_dir=training_args.cache_dir,
        model_max_length=training_args.model_max_length,
        padding_side="right",
        use_fast=training_args.fast_tokenizer,
        trust_remote_code=True,
        use_auth_token=True,
    )

    if 'mpt' in model_args.model_name_or_path.lower():
        config = transformers.AutoConfig.from_pretrained(
            model_args.model_name_or_path,
            trust_remote_code=True,
        )
        if model_args.use_flash_attention:
            config.attn_config['attn_impl'] = model_args.flash_attention_version
        config.update({"max_seq_len": max(config.max_seq_len, training_args.model_max_length)})
        setattr(config, 'hidden_size', config.d_model)

        model = transformers.AutoModelForCausalLM.from_pretrained(
            model_args.model_name_or_path,
            config=config,
            trust_remote_code=True,
            use_auth_token=True,
        )
    else:
        model = transformers.AutoModelForCausalLM.from_pretrained(
            model_args.model_name_or_path,
            cache_dir=training_args.cache_dir,
            trust_remote_code=True,
            use_auth_token=True,
        )
        if hasattr(model.config, "max_position_embeddings") and training_args.model_max_length > model.config.max_position_embeddings:
            max_positions = training_args.model_max_length 
            for m in model.modules():
                if "GPTNeoXAttention" in str(type(m)) or "GPT2Attention" in str(type(m)):
                    m.register_buffer(
                        "bias",
                        torch.tril(torch.ones((max_positions, max_positions), dtype=torch.bool)).view(
                            1, 1, max_positions, max_positions
                        ),
                    )
            if model.config.model_type == "gpt2":
                old_wpe = model.transformer.wpe
                model.transformer.wpe = torch.nn.Embedding(max_positions, model.config.n_embd)
                with torch.no_grad():
                    # copy old positional embeddings
                    model.transformer.wpe.weight[:old_wpe.weight.shape[0], :] = old_wpe.weight

            model.config.max_position_embeddings = max_positions

        if model_args.use_flash_attention:
            print('use flash attention')
            from as_lm_pretrain.models.utils import apply_flash_attention
            apply_flash_attention(model, version=model_args.flash_attention_version)

    if model_args.use_lora:
        from peft import LoraConfig, get_peft_model, TaskType

        model_type = str(type(model))
        if 'GPTNeoXForCausalLM' in model_type:
            r = model_args.lora_r * 3
            target_modules = ["query_key_value"]
        elif 'MPTForCausalLM' in model_type: 
            r = model_args.lora_r * 3
            target_modules = ["Wqkv"]
        else:
            r = model_args.lora_r
            target_modules = ["q_proj", "v_proj"]
        lora_config = LoraConfig(
            r=r,
            lora_alpha=model_args.lora_alpha,
            target_modules=target_modules,
            lora_dropout=model_args.lora_dropout,
            bias="none",
            task_type="CAUSAL_LM",
        )
        model = get_peft_model(model, lora_config)

    if 'mpt' not in model_args.model_name_or_path.lower():
        if tokenizer.pad_token is None:
            smart_tokenizer_and_embedding_resize(
                special_tokens_dict=dict(pad_token=DEFAULT_PAD_TOKEN),
                tokenizer=tokenizer,
                model=model,
            )
    # if "llama" in model_args.model_name_or_path:
    if 'LlamaForCausalLM' in str(type(model)):
        tokenizer.add_special_tokens(
            {
                "eos_token": DEFAULT_EOS_TOKEN,
                "bos_token": DEFAULT_BOS_TOKEN,
                "unk_token": DEFAULT_UNK_TOKEN,
            }
        )
    tokenizer.pad_token = tokenizer.eos_token

    data_module = make_supervised_data_module(tokenizer=tokenizer, data_args=data_args)
    trainer = Trainer(model=model, tokenizer=tokenizer, args=training_args, **data_module)
    if list(pathlib.Path(training_args.output_dir).glob("checkpoint-*")):
        trainer.train(resume_from_checkpoint=True)
    else:
        trainer.train()
    trainer.save_state()
    safe_save_model_for_hf_trainer(trainer=trainer, output_dir=training_args.output_dir)


if __name__ == "__main__":
    train()
