# ulimit -n 4096
# export MLFLOW_TRACKING_URI="https://mlflow-dev-external.shared.asapp.dev/"
# export MLFLOW_EXPERIMENT_NAME=asapp-alpaca
# export LOGNAME=fwu

function train_model {
master_port=`shuf -i 20000-35000 -n 1`
deepspeed --num_gpus=$ngpu --master_port $master_port train.py \
    --deepspeed $deepspeed_config \
    --model_name_or_path $model_path \
    --data_path $data_path \
    --output_dir ${save_root}/${run_name} \
    --num_train_epochs 3 \
    --per_device_train_batch_size $batch_size \
    --per_device_eval_batch_size $batch_size \
    --gradient_accumulation_steps $gradient_accumulation_steps \
    --evaluation_strategy "no" \
    --save_strategy "epoch" \
    --learning_rate $lr \
    --weight_decay 0. \
    --warmup_ratio 0.03 \
    --lr_scheduler_type "cosine" \
    --logging_steps 1 \
    --report_to tensorboard \
    --model_max_length $model_max_length \
    $@
    # --tf32 True \
    # --fsdp "full_shard auto_wrap" \
    # --fsdp_transformer_layer_cls_to_wrap $transformer_layer \
    # --save_steps 2000 \
    # --save_total_limit 1 \
}

ngpu=3
save_root=models/2310
model_max_length=2048

lr=2e-5
batch_size=1
gradient_accumulation_steps=$((16 / $batch_size))


batch_size=1
gradient_accumulation_steps=$((16 / $batch_size))
model_max_length=2048
data_path=alpaca_data.json
model_path=facebook/opt-6.7b
deepspeed_config=`realpath ds_configs/ds-zero3-decay.json`
run_name=${model_path/\//-}-${data_name}-lr${lr}-bs${batch_size}-gac${gradient_accumulation_steps}-max_length${model_max_length}-zero3
train_model --bf16 True --use_lora True 
