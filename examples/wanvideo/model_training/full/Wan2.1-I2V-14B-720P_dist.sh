#!/bin/bash
set -euo pipefail

NODE_RANK="$1"
MASTER_ADDR="$2"
MASTER_PORT="$3"

NUM_NODES="${NNODES:-2}"
NUM_GPUS="${NUM_GPUS:-8}"

deepspeed \
  --hostfile ./examples/wanvideo/model_training/full/hostfile \
  --no_ssh \
  --num_nodes "${NUM_NODES}" \
  --num_gpus "${NUM_GPUS}" \
  --node_rank "${NODE_RANK}" \
  --master_addr "${MASTER_ADDR}" \
  --master_port "${MASTER_PORT}" \
  examples/wanvideo/model_training/train.py \
    --dataset_base_path data/example_video_dataset \
    --dataset_metadata_path data/example_video_dataset/metadata.csv \
    --height 720 \
    --width 1280 \
    --num_frames 49 \
    --dataset_num_workers 8 \
    --dataset_repeat 100 \
    --model_id_with_origin_paths "Wan-AI/Wan2.1-I2V-14B-720P:diffusion_pytorch_model*.safetensors,Wan-AI/Wan2.1-I2V-14B-720P:models_t5_umt5-xxl-enc-bf16.pth,Wan-AI/Wan2.1-I2V-14B-720P:Wan2.1_VAE.pth,Wan-AI/Wan2.1-I2V-14B-720P:models_clip_open-clip-xlm-roberta-large-vit-huge-14.pth" \
    --learning_rate 1e-5 \
    --num_epochs 10 \
    --remove_prefix_in_ckpt "pipe.dit." \
    --output_path "./models/train/Wan2.1-I2V-14B-720P_full" \
    --trainable_models "dit" \
    --extra_inputs "input_image" \
    --use_gradient_checkpointing \
    --deepspeed \
    --deepspeed_config examples/wanvideo/model_training/full/deepspeed_stage1_config.json

