# source /aigc_cfs_2/neoshang/software/anaconda3/bin/activate diffusionsdf

json_path=/aigc_cfs_2/neoshang/data/data_list/20240520/part12_0_120k_color_only.json
json_save_path=/aigc_cfs_2/neoshang/data/data_list/20240520/part12_0_120k_color_only_h5.json

cd /aigc_cfs_2/neoshang/code/diffusers_triplane/datasets_diffusion

python -m dataset.h5 -i $json_path -o $json_save_path \
        -d /aigc_cfs_5/neoshang/h5/1 \
        --n_workers 128 \
        --chunk_size_n_views 1 \
        --images_only