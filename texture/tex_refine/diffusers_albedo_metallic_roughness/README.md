# Env

1. Install requirements first
```bash
pip install -r requirements.txt
```


# Download ptetrain models
- Download 'facebook/dinov2-large' from huggingface and replace 'preprocessor_config.json' with 'pretrain_ckpts/dinov2-large/preprocessor_config.json'

- Download 'laion/CLIP-ViT-bigG-14-laion2B-39B-b160k' from huggingface

- Download vae pretrain ckpt
- Download diffusion pretrain ckpt
- Download RMBG-1.4 ckpt from huggingface

# Logs
We use wandb offline as logger here. you need to change the wandb private key to yours. in training scripts.
```python
os.environ["WANDB_API_KEY"] = "your_wandb_key" #### change to your own wandb key
```

# Data
- Images, pcd and json data sample, please refer './data' folder

- The dataloader will copy the origin data in json to the local machine during training, for speedup. If you don't need it just feel free to close it.

# Train
```bash
albedo 训练： bash sh/train_img2img_albedo_6views_512.sh
roughness/metallic 训练： bash sh/train_img2img_material.sh
```

# Test
```python
cd examples/modality_transfer
python test_img2img_albedo_6views_real_img_512_single_img_input.py
python test_img2img_metallic_roughness_realworld_single.py
