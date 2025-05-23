export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
CUDA_VISIBLE_DEVICES="0" 
python train/liutao/train_flux_lora.py \
  --pretrained_text_encoder_path models/FLUX/FLUX.1-dev/text_encoder/model.safetensors \
  --pretrained_text_encoder_2_path models/FLUX/FLUX.1-dev/text_encoder_2 \
  --pretrained_dit_path models/FLUX/FLUX.1-dev/flux1-dev.safetensors \
  --pretrained_vae_path models/FLUX/FLUX.1-dev/ae.safetensors \
  --dataset_path train/liutao/data/liutao \
  --output_path ./models/lora/liutao \
  --max_epochs 1 \
  --steps_per_epoch 100 \
  --height 1024 \
  --width 1024 \
  --center_crop \
  --precision "bf16" \
  --learning_rate 1e-4 \
  --lora_rank 4 \
  --lora_alpha 8 \
  --use_gradient_checkpointing \
  --align_to_opensource_format \
  --quantize "float8_e4m3fn"