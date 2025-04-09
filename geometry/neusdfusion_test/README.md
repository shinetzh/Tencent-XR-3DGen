
# NeuSDFusion: A Spatial-Aware Generative Model for 3D Shape Completion, Reconstruction, and Generation

### The European Conference on Computer Vision (ECCV), 2024

### [Project page](https://weizheliu.github.io/NeuSDFusion/) | [Paper](https://weizheliu.github.io/NeuSDFusion/static/pdfs/neusdfusion.pdf)

This repo contains the official implementation for the paper "NeuSDFusion: A Spatial-Aware Generative Model for 3D Shape Completion, Reconstruction, and Generation".

## Environment
We provide the requirements.txt for environment setup.
```
  pip install -r requirments.txt
```

## Training and Testing
We use pix3d dataset as example to show how to train and test our model

### Training VAE
```python
python train.py \
  --exp_dir config/paper_stage1_pix3d \
  -b 10 \
  -w 8
```

### Testing VAE
```python
python test.py \
  --exp_dir config/paper_stage1_pix3d \
  --num_samples 100000
```

### Training Diffusion

- save vae latent
- config stage2 diffusion specs.json

```python
python train.py \
    -e config/paper_stage2_diff_pix3d \
    -b 256 \
    -w 8
```
### Testing Diffusion
```python
python test.py \
  --exp_dir config/paper_stage2_diff_pix3d/ \
  --num_samples 10000
```

## Citation

If you find our code or paper helps, please consider citing:
```
@inproceedings{cui2024neusdfusion,
  title={Neusdfusion: A spatial-aware generative model for 3d shape completion, reconstruction, and generation},
  author={Cui, Ruikai and Liu, Weizhe and Sun, Weixuan and Wang, Senbo and Shang, Taizhang and Li, Yang and Song, Xibin and Yan, Han and Wu, Zhennan and Chen, Shenzhou and others},
  booktitle={European Conference on Computer Vision},
  year={2024},
  organization={Springer}
}    
```