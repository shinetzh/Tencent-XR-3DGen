cd /apdcephfs/private_xiaqiangdai/workspace/RAFT-Stereo/
python3    demo.py --restore_ckpt models/raftstereo-eth3d.pth --corr_implementation reg  --mixed_precision  -l=/apdcephfs_cq2/share_1615605/xiaqiangdai/datasets_stereo/Middlebury/MiddEval3/testF/*/im0.png  \
-r=/apdcephfs_cq2/share_1615605/xiaqiangdai/datasets_stereo/Middlebury/MiddEval3/testF/*/im1.png 