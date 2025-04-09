cd /apdcephfs/private_xiaqiangdai/workspace/RAFT-Stereo/
python3 train_stereo.py --train_datasets middlebury_2014 --num_steps 4000 --image_size 384 1000  --shared_backbone  --batch_size 2 
--train_iters 22 --valid_iters 32 --spatial_scale -0.2 0.4 --saturation_range 0 1.4 --n_downsample 2  --mixed_precision