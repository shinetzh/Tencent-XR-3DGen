export PYTHONPATH=$PWD:$PYTHONPATH

# nccl settings
#export NCCL_DEBUG=INFO
export NCCL_SOCKET_IFNAME=eth0
export NCCL_IB_GID_INDEX=3
export NCCL_IB_DISABLE=0
export NCCL_NET_GDR_LEVEL=2
export NCCL_IB_QPS_PER_CONNECTION=4
export NCCL_IB_TC=160
export NCCL_IB_TIMEOUT=22
# export NCCL_P2P_DISABLE=1

export NCCL_P2P_LEVEL=NVL
# export NCCL_DEBUG=trace
export CUDA_LAUNCH_BLOCKING=1
export TORCH_NCCL_ASYNC_ERROR_HANDLING=1
export TORCH_NCCL_DUMP_ON_TIMEOUT=true
export TORCH_NCCL_TRACE_BUFFER_SIZE=2000

# node settings
MASTER_ADDR=${1:-localhost}
NNODES=${2:-1}
NODE_RANK=${3:-0}
MASTER_PORT=6000
GPUS_PER_NODE=8
DISTRIBUTED_ARGS="--nproc_per_node $GPUS_PER_NODE --nnodes $NNODES --node_rank $NODE_RANK --master_addr $MASTER_ADDR --master_port $MASTER_PORT"

echo $MASTER_ADDR
echo $NNODES
echo $NODE_RANK

training_name=1view_gray_2048_flow ## training config name
batch_size=2
gradient_accumulation_steps=1
learning_rate=7e-5
output_dir=configs/$training_name
pretrained_model_name_or_path=./$output_dir
tracker_project_name=$training_name
log_dir=$output_dir/logs


#### train image to 3D with 1 image condition
CMD="torchrun $DISTRIBUTED_ARGS \
    scripts/train_mmdit_image23D_flow_1view.py \
    --output_dir=$output_dir \
    --pretrained_model_name_or_path=$pretrained_model_name_or_path \
    --validation_images_dir="/data/validation/images" \
    --tracker_project_name=$tracker_project_name \
    --use_ema \
    --do_classifier_free_guidance \
    --train_batch_size=$batch_size --gradient_accumulation_steps=$gradient_accumulation_steps  --gradient_checkpointing \
    --num_train_epochs=20 \
    --validation_epochs=1 \
    --checkpointing_steps=3000 --checkpoints_total_limit=15 \
    --lr_scheduler "cosine" --learning_rate=$learning_rate --lr_warmup_steps=1000 --lr_num_cycles=4 \
    --max_grad_norm=1 \
    --dataloader_num_workers 8 \
    --mixed_precision="bf16" \
    --drop_condition_prob 0.1 \
    --report_to="tensorboard"
    "


# #### train image to 3D with 1 image condition
# CMD="torchrun $DISTRIBUTED_ARGS \
#     scripts/train_mmdit_image23D_flow_4views.py \
#     --output_dir=$output_dir \
#     --pretrained_model_name_or_path=$pretrained_model_name_or_path \
#     --validation_images_dir="/data/validation/images" \
#     --tracker_project_name=$tracker_project_name \
#     --use_ema \
#     --do_classifier_free_guidance \
#     --train_batch_size=$batch_size --gradient_accumulation_steps=$gradient_accumulation_steps  --gradient_checkpointing \
#     --num_train_epochs=20 \
#     --validation_epochs=1 \
#     --checkpointing_steps=3000 --checkpoints_total_limit=15 \
#     --lr_scheduler "cosine" --learning_rate=$learning_rate --lr_warmup_steps=1000 --lr_num_cycles=4 \
#     --max_grad_norm=1 \
#     --dataloader_num_workers 8 \
#     --mixed_precision="bf16" \
#     --drop_condition_prob 0.1 \
#     --report_to="tensorboard"
#     "


echo $CMD 2>&1 | tee $log_dir/log.log
eval $CMD 2>&1 | tee -a $log_dir/log.log
