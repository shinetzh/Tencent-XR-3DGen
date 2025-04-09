# <div align = "center">‌Tencent XR 3DGen Suite: Integrated Framework for High-Quality 3D Shape Generation, Texture Synthesis, and Character Customization</div>

<div align="center">
<a href="https://arxiv.org/abs/2502.14247"><img src="https://img.shields.io/badge/ArXiv-2502.14247-004088.svg"/></a>
<a href=https://huggingface.co/spaces/peterji/PandoraX><img src=https://img.shields.io/badge/%F0%9F%A4%97%20Demo-276cb4.svg></a>
<a href=https://huggingface.co/peterji/Tencent_XR_3DGen><img src=https://img.shields.io/badge/%F0%9F%A4%97%20Models-d96902.svg></a>
<!-- <a href="https://youtu.be/guFg_Ppt1Ag">
<img alt="Youtube" src="https://img.shields.io/badge/Video-Youtube-red"/>
</a> -->
<!-- <a ><img alt="PRs-Welcome" src="https://img.shields.io/badge/PRs-Welcome-red" /></a> -->
<a href="https://github.com/Tencent/Tencent-XR-3DGen/stargazers">
<img alt="stars" src="https://img.shields.io/github/stars/Tencent/Tencent-XR-3DGen" />
</a>
<a href="https://github.com/Tencent/Tencent-XR-3DGen/network/members">
<img alt="FORK" src="https://img.shields.io/github/forks/MAVIS-SLAM/ORB_SLAM3_MULTI?color=FF8000" />
</a>
<a href="https://github.com/Tencent/Tencent-XR-3DGen/issues">
<img alt="Issues" src="https://img.shields.io/github/issues/Tencent/Tencent-XR-3DGen?color=0088ff"/>
</a>
</div>

<p align="center">
  <img src="assets/3a92f986-096a-4ec3-a03b-8374ca5bffa1.gif" width="16%" />
  <img src="assets/a7ca5ffb-8d33-46ee-afb8-0c80e1169deb.gif" width="16%" />
  <img src="assets/c23cc595-b4b5-4563-a987-3c5174797daf.gif" width="16%" />
  <img src="assets/a4e1428b-9a89-43f3-b8fb-5440983d6379.gif" width="16%" />
  <img src="assets/c3e91666-994f-431d-a184-e89a6442c661.gif" width="16%" />
  <img src="assets/48e06e88-6a1e-4480-a7eb-f98ba79d328e.gif" width="16%" />
</p>


## Overview

This is an all-in-one code release for 3D generation pipeline developed by Tencent XR Vision Labs. It contains following submodules, including training and inference codes:

```
PandoraX
├── geometry
│   ├── main_pipeline
│   │   ├── vae             (geometry_vae)
│   │   └── diffusion       (geometry_dit)
│   ├── am_generation       (ArtistCreatedMeshes)
│   ├── neusdfusion         (NeuSDFusion_release)
│   └── isosurface          (sparse_mc)
├── texture
│   ├── main_pipeline       (texture_generation)
│   ├── geo2rgb             (multiview_geo2rgb)
│   ├── rgb2pbr             (multiview_rgbpbr)
│   └── render_bake         (render-bake_utilities)
├── data
│   ├── main_data_pipeline  (data_preparation)
│   └── openvdb_interface   (python-openvdb)
├── character
│   ├── main                (character_generation)
│   └── phy_cage            (PhyCAGE_release)
├── misc
│   ├── raft_stereo         (RAFT-Stereo_training)
│   ├── vs_tool             (VSTool)
│   ├── quad_remesh         (quad_remesh_utils)
│   └── texture_utils       (frontal_image_generation_4_texture_gen)

```

## Dependencies
dependencies for submodules are zipped under `requirements_txt.zip`

## Model weights
model weights and tech report can be downloaded from https://drive.google.com/drive/folders/1NgtWeouNiM-G5VtZMQRF10f2ymNzLsdn

## Inference

1. After downloading the weights, please update the "pretrained_weights_path" in ./geometry/main_pipeline/diffusion/utils/rmbg1_4.py and ./texture/tex_refine/sam_preprocess/run_sam.py

2. python test_inference.py --image_path path of the input image --save_dir path of the save dir --weights_path path of the pretrained weights

## Citation

@article{yang2025pandora3d,

  title={Pandora3D: A Comprehensive Framework for High-Quality 3D Shape and Texture Generation},

  author={Yang, Jiayu and Shang, Taizhang and Sun, Weixuan and Song, Xibin and Chen, Ziang and Wang, Senbo and Chen, Shenzhou and Liu, Weizhe and Li, Hongdong and Ji, Pan},

  journal={arXiv preprint arXiv:2502.14247},

  year={2025}

}

## License

Tencent XR 3DGen is licensed under the MIT License but with additional restrictions in accordance with company policies. These include prohibitions on using the software for harmful or discriminatory purposes, or within the European Union. Please refer to the LICENSE file for full details on the terms and conditions.
