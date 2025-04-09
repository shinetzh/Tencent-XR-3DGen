# 网络名称,同目录名称,需要模型审视修改
Network="zero123plus_v1"

#训练开始时间，不需要修改
start_time=$(date +%s)
echo "start_time: ${start_time}"

# config_path=$1
# accelerate launch --config_file $1 \

# config_path=$1
# accelerate launch --config_file $1 \
# ASCEND_RT_VISIBLE_DEVICES=0 \
# ASCEND_RT_VISIBLE_DEVICES=0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15 \
# CUDA_VISIBLE_DEVICES=0 \
# ASCEND_RT_VISIBLE_DEVICES=0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15 \
# accelerate launch --num_processes 1 --main_process_port 31888 \
# ASCEND_RT_VISIBLE_DEVICES=8,9,10,11,12,13,14,15 \
# ASCEND_RT_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 \
# accelerate launch --multi_gpu --num_processes 1 --main_process_port 31268 \
# ASCEND_RT_VISIBLE_DEVICES=0,1,2,3,4,5,6,7,8,9,10,11,12,1 \
# accelerate launch --num_processes 16 --main_process_port 36298 \
# ASCEND_RT_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 \
# accelerate launch --num_processes 8 --main_process_port 36698 \
# accelerate launch --multi_gpu --num_processes=16 --num_machines=1 --machine_rank=0 --main_process_ip=***.***.***.*** --main_process_port=31968 \
ASCEND_RT_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 \
accelerate launch --num_processes 8 --main_process_port 36668 \
    examples/modality_transfer/train_img2img_delight_single_full_light.py \
    --output_dir="/aigc_cfs_4/xibin/code/diffusers_triplane_models/delight_models_full_light_1024" \
    --pretrained_model_name_or_path="/aigc_cfs_4/xibin/code/diffusers_triplane_models/delight_models_full_light_1024" \
    --validation_images_dir="/aigc_cfs_2/neoshang/code/diffusers_triplane/data/validation" \
    --tracker_project_name "rgb2norm_v1" \
    --use_ema \
    --do_classifier_free_guidance \
    --prediction_type "v_prediction" \
    --train_batch_size=2 --gradient_accumulation_steps=6 --gradient_checkpointing \
    --num_train_epochs=100 \
    --validation_epochs=1 \
    --checkpointing_steps=1000 --checkpoints_total_limit=200 \
    --lr_scheduler "cosine" --learning_rate=5e-05 --lr_warmup_steps=1000 \
    --max_grad_norm=1 \
    --dataloader_num_workers 8 \
    --mixed_precision="fp16" \
    --snr_gamma 5.0 \
    --drop_condition_prob 0.08 \
    --report_to="tensorboard" \
    --resume_from_checkpoint "latest"
wait


#训练结束时间，不需要修改
end_time=$(date +%s)
e2e_time=$(($end_time - $start_time))

#结果打印，不需要修改
echo "------------------ Final result ------------------"

echo "E2E Training Duration sec : $e2e_time"