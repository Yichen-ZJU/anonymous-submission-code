#!/bin/bash
# DataValve: Selecting-while-Training via Utility-Gated Routing
# Stage: Full training pipeline for LLaVA-1.5-7B-LoRA

set -e
export DS_SKIP_CUDA_CHECK=1

# ==================== CUDA / NCCL ====================
export CUDA_HOME=${CUDA_HOME:-/usr/local/cuda-12.1}
export PATH=$CUDA_HOME/bin:$PATH
export LD_LIBRARY_PATH=$CUDA_HOME/lib64:/usr/lib/x86_64-linux-gnu:$LD_LIBRARY_PATH
export CUDA_LAUNCH_BLOCKING=${CUDA_LAUNCH_BLOCKING:-0}

export NCCL_TIMEOUT=3600
export NCCL_SOCKET_IFNAME=${NCCL_SOCKET_IFNAME:-lo}
export NCCL_IB_DISABLE=${NCCL_IB_DISABLE:-1}
export NCCL_P2P_DISABLE=${NCCL_P2P_DISABLE:-1}
export NCCL_TREE_THRESHOLD=0
export NCCL_ASYNC_ERROR_HANDLING=1
export TORCH_NCCL_ASYNC_ERROR_HANDLING=1

# ==================== Paths ====================
: ${DATA_DIR:="./data"}
: ${OUTPUT_DIR:="./checkpoints/datavalve_$(date +%Y%m%d_%H%M%S)"}
: ${MODEL_DIR:="./checkpoints"}
: ${DEEPSPEED_CONFIG:="./LLaVA/scripts/zero3.json"}

TRAIN_DATA="${DATA_DIR}/training_set.json"
GOLDEN_DATA="${DATA_DIR}/golden_set.json"
CLIP_FEATURES="${DATA_DIR}/scores/llava_clip_feature.pt"
IMAGE_FOLDER="${DATA_DIR}"

BASE_MODEL="${MODEL_DIR}/vicuna-7b-v1.5"
VISION_TOWER="${MODEL_DIR}/clip-vit-large-patch14/"
PRETRAIN_MM_MLP_ADAPTER="${MODEL_DIR}/llava-v1.5-mlp2x-336px-pretrain-vicuna-7b-v1.5/mm_projector.bin"

# ==================== Core Hyperparams ====================
TARGET_RATIO=0.208
WARMUP_RATIO=0.03
FINAL_BATCH_SIZE=8
SUPER_BATCH_MULTIPLIER=6
ROUTER_UPDATE_FREQ=10

# Loss
LAMBDA_REINFORCE=1.0

# Router
ROUTER_HIDDEN_DIM=256
ROUTER_LR=1.5e-4
ROUTER_BIAS_INIT=0.5

# LLaVA
LLAVA_LR=2e-4
NUM_EPOCHS=1
LORA_R=128
LORA_ALPHA=256

# Training
GOLDEN_BATCH_SIZE=32
GRAD_ACCUM=1
SAVE_STEPS=500
LOGGING_STEPS=10
WARMUP_STEPS=100
SEED=42

GPU_IDS=${GPU_IDS:-"0,1,2,3"}
MASTER_PORT=${MASTER_PORT:-12345}

mkdir -p "$OUTPUT_DIR"

deepspeed --include "localhost:${GPU_IDS}" --master_port "$MASTER_PORT" datavalve/train.py \
    --deepspeed "$DEEPSPEED_CONFIG" \
    --model_name_or_path "$BASE_MODEL" \
    --version v1_5 \
    --data_path "$TRAIN_DATA" \
    --golden_data_path "$GOLDEN_DATA" \
    --clip_features_path "$CLIP_FEATURES" \
    --image_folder "$IMAGE_FOLDER" \
    --vision_tower "$VISION_TOWER" \
    --pretrain_mm_mlp_adapter "$PRETRAIN_MM_MLP_ADAPTER" \
    --mm_vision_select_layer -2 \
    --mm_use_im_start_end False \
    --mm_use_im_patch_token False \
    --bf16 True \
    --mm_projector_type mlp2x_gelu \
    --output_dir "$OUTPUT_DIR" \
    --report_to none \
    --num_train_epochs "$NUM_EPOCHS" \
    --per_device_train_batch_size "$FINAL_BATCH_SIZE" \
    --gradient_accumulation_steps "$GRAD_ACCUM" \
    --evaluation_strategy "no" \
    --save_strategy "steps" \
    --save_steps "$SAVE_STEPS" \
    --save_total_limit 2 \
    --learning_rate "$LLAVA_LR" \
    --router_lr "$ROUTER_LR" \
    --weight_decay 0. \
    --warmup_steps "$WARMUP_STEPS" \
    --lr_scheduler_type "cosine" \
    --logging_steps "$LOGGING_STEPS" \
    --tf32 True \
    --model_max_length 2048 \
    --gradient_checkpointing True \
    --dataloader_num_workers 0 \
    --lazy_preprocess True \
    --group_by_modality_length True \
    --target_ratio "$TARGET_RATIO" \
    --warmup_ratio "$WARMUP_RATIO" \
    --super_batch_multiplier "$SUPER_BATCH_MULTIPLIER" \
    --router_update_freq "$ROUTER_UPDATE_FREQ" \
    --lambda_reinforce "$LAMBDA_REINFORCE" \
    --router_hidden_dim "$ROUTER_HIDDEN_DIM" \
    --router_bias_init "$ROUTER_BIAS_INIT" \
    --lora_enable True \
    --lora_r "$LORA_R" \
    --lora_alpha "$LORA_ALPHA" \
    --lora_dropout 0.05 \
    --golden_batch_size "$GOLDEN_BATCH_SIZE" \
    --seed "$SEED"

echo "Training complete. Checkpoint saved to $OUTPUT_DIR"
