# Improve sd1.5 inpainting  

<div style="display: flex; justify-content: center; align-items: center;">
  <a href="https://sjbfwnygg89.sg.larksuite.com/wiki/Qq8xwqC5yivuWJkIF81lK4Fig0c" style="margin: 0 2px;">
    <img src='https://img.shields.io/badge/arXiv-2407.15886-red?style=flat&logo=arXiv&logoColor=red' alt='arxiv'>
  </a>
</div>



This experiment aims to enhance TryOnNet by leveraging Stable Diffusion 1.5 (SD1.5) as the base model, exploring two experimental approaches to improve garment try-on quality and realism.

## Cases of experiment:
### Case 1: SD1.5 Inpainting Baseline + Extra_dim + Ip_adapter
This approach builds on the foundational SD1.5 Inpainting model, further improving its performance using:

** Extra_dim: Enhances dimensional awareness for better garment alignment (add 1 garment channel).
** Ip_adapter: A specialized module to transfer garment style correctly and learn high-level information effectively, ensuring garments fit naturally and adhere to subject-specific contours.

### Case 2: SD1.5 Inpainting + GarmentNet
This approach incorporates GarmentNet, a model designed to explicitly learn garment structures and textures. By combining GarmentNet with SD1.5 Inpainting, the system achieves:
** Improved garment texture preservation.
** More accurate mapping of garment features, such as wrinkles, patterns, and edges.

## Structure 
### 1. Dataset preparation
```
├── VITON-HD
|   ├── test_pairs_unpaired.txt
│   ├── test
|   |   ├── image
│   │   │   ├── [000006_00.jpg | 000008_00.jpg | ...]
│   │   ├── cloth
│   │   │   ├── [000006_00.jpg | 000008_00.jpg | ...]
│   │   ├── agnostic-mask
│   │   │   ├── [000006_00_mask.png | 000008_00.png | ...]
...
```

### 2. Experiment structure
```
├── hungvo_experiment
|   |      
|   ├── garment_adapter
|   |   ├── attention_garm.py
│   │   ├── attention_processor.py
|   |   ├── transformer_garm_2d.py
|   |   ├── unet_garm_2d_blocks.py
│   │   |__ unet_garm_2d_condition.py
|   |
|   |
|   ├── ip_adapter
|   |   ├── attention_preprocessor.py
│   │   |__ resampler.py
│   |
|   |
|   ├── pipeline
│   |   ├── pipeline_stable_diffusion_inpaint_tryon_ref.py
│   │   |__ pipeline_stable_diffusion_inpaint_tryon.py
│   |
|   |
|   ├── README.md
│   ├── requirements.txt
|   ├── run.
|   |
|   |
|   ├── test_stablediffusion_inpaint_with_cloth_channel_ref_without_cloth_latent.py
|   ├── test_stablediffusion_inpaint_with_cloth_channel_ref.py
|   ├── test_stablediffusion_inpaint_with_cloth_channel.py



```


## Download Resources 

### Dataset  
Download the dataset using the following command:  
```bash
wget https://drive.google.com/drive/u/3/folders/1Vv--aWuDvz7Nd6TaXrZjRmqJ7VoKD41m
```

### IP Adapter  
Download the IP adapter using the following command:  
```bash
cd adapter
wget https://huggingface.co/h94/IP-Adapter/resolve/main/models/ip-adapter-plus_sd15.bin
```



## Training code
1. Sd1.5 inpainting: 
```
  python train.py  --clip_penultimate=False --train_batch_size=1 --gradient_accumulation_steps=8 --max_train_steps=1000000 --learning_rate=1e-5 --weight_decay=0.01 --lr_scheduler="constant" --num_warmup_steps=2000 --output_dir="fnc_valo" --checkpointing_steps=10
```
2. Sd1.5 inpainting + garment_net:
```
    accelerate launch train.py \
  --pretrained_model_name_or_path="/path_to/stable-diffusion-v1-5/" \
  --pretrained_vae_model_path="/path_to/sd-vae-ft-mse/" \
  --pretrained_adapter_model_path="/path_to/IP-Adapter/ip-adapter-plus_sd15.bin" \
  --image_encoder_path="/path_to/h94/IP-Adapter/models/image_encoder" \
  --dataset_json_path="/path_to/IGPair.json" \
  --clip_penultimate=False \
  --train_batch_size=5 \
  --gradient_accumulation_steps=1 \
  --max_train_steps=1000000 \
  --learning_rate=1e-5 \
  --weight_decay=0.01 \
  --lr_scheduler="constant" --num_warmup_steps=2000 \
  --output_dir="/save_path" \
  --checkpointing_steps=10000
```