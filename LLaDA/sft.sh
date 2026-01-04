
set -euo pipefail

SCRIPT="sft.py"
MODEL_DIR="input_model"
DATA_FILE="sft_data.txt"
OUTPUT_DIR="output_model"
DS_CONFIG="ds_config.json"

NPROC=${NPROC:-2} 
MASTER_ADDR=${MASTER_ADDR:-127.0.0.1}
MASTER_PORT=${MASTER_PORT:-29551}

BATCH=${BATCH:-1}
GRAD_ACCUM=${GRAD_ACCUM:-2}
LR=${LR:-1e-5}
EPOCHS=${EPOCHS:-50}
MAXLEN=${MAXLEN:-128}
NUM_WORKERS=${NUM_WORKERS:-4}

[[ -f "$SCRIPT" ]] || { echo "找不到训练脚本: $SCRIPT"; exit 1; }
[[ -d "$MODEL_DIR" ]] || { echo "找不到模型目录: $MODEL_DIR"; exit 1; }
[[ -f "$DATA_FILE" ]] || { echo "找不到数据文件: $DATA_FILE"; exit 1; }
[[ -f "$DS_CONFIG" ]] || { echo "找不到 DeepSpeed 配置: $DS_CONFIG"; exit 1; }

TIMESTAMP=$(date +%Y%m%d_%H%M%S)
LOG_DIR="${OUTPUT_DIR}/logs"; mkdir -p "$OUTPUT_DIR" "$LOG_DIR"
LOG_FILE="${LOG_DIR}/train_${TIMESTAMP}.log"

echo "日志写入: $LOG_FILE"

set -x
export CUDA_VISIBLE_DEVICES=2,3

torchrun --nproc_per_node="${NPROC}" --master_addr="${MASTER_ADDR}" --master_port="${MASTER_PORT}" \
  "$SCRIPT" \
  --model_name_or_path "$MODEL_DIR" \
  --dataset_txt_path "$DATA_FILE" \
  --output_dir "$OUTPUT_DIR" \
  --deepspeed_config "$DS_CONFIG" \
  --per_device_train_batch_size "${BATCH}" \
  --gradient_accumulation_steps "${GRAD_ACCUM}" \
  --learning_rate "${LR}" \
  --num_train_epochs "${EPOCHS}" \
  --max_seq_length "${MAXLEN}" \
  --num_workers "${NUM_WORKERS}" \
  --ckpt_strategy "whole_layer" \
  2>&1 | tee "$LOG_FILE"