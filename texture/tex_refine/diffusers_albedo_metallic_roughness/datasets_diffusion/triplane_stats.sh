# source /apdcephfs/private_neoshang/software/envs/diffusers/bin/activate
source /apdcephfs/private_neoshang/software/anaconda3/bin/activate diffusionsdf

CUDA_VISIBLE_DEVICES=4 \
python datasets_diffusion/triplane_stats.py \
    --config_dir "/aigc_cfs_2/neoshang/code/diffusers_triplane/configs/triplane_conditional_sdfcolor_objaverse_kl_v0.0.0" \
    --key "latent_modulation"