
set -euo pipefail


SCRIPT_PATH="pretrain.py"


MODEL_DIR="input_model"

DATA_FILE="origin_prompts.txt"

OUTPUT_DIR="output_dir"

DS_CONFIG="ds_config.json"


NPROC_PER_NODE=${NPROC_PER_NODE:-2}

MASTER_ADDR=${MASTER_ADDR:-"127.0.0.1"}

MASTER_PORT=${MASTER_PORT:-$(shuf -i 10000-65535 -n 1)}


BATCH_SIZE_PER_GPU=${BATCH_SIZE_PER_GPU:-1}


LEARNING_RATE=${LEARNING_RATE:-1e-5}


EPOCHS=${EPOCHS:-3}
BLOCK_SIZE=${BLOCK_SIZE:-4096}

NUM_WORKERS=${NUM_WORKERS:-4}

CKPT_STRATEGY=${CKPT_STRATEGY:-"whole_layer"}



[[ -f "$SCRIPT_PATH" ]] || { echo "Error: Training script not found: $SCRIPT_PATH"; exit 1; }
[[ -d "$MODEL_DIR" ]]   || { echo "Error: Model directory not found: $MODEL_DIR"; exit 1; }
[[ -f "$DATA_FILE" ]]   || { echo "Error: Data file not found: $DATA_FILE"; exit 1; }
[[ -f "$DS_CONFIG" ]]   || { echo "Error: DeepSpeed config file not found: $DS_CONFIG"; exit 1; }


TIMESTAMP=$(date +%Y%m%d_%H%M%S)
LOG_DIR="${OUTPUT_DIR}/logs"
mkdir -p "$OUTPUT_DIR" "$LOG_DIR"
LOG_FILE="${LOG_DIR}/pretrain_${TIMESTAMP}.log"






set -x
export CUDA_VISIBLE_DEVICES=2,3

torchrun --nproc_per_node="${NPROC_PER_NODE}" --master_addr="${MASTER_ADDR}" --master_port="${MASTER_PORT}" \
    "$SCRIPT_PATH" \
    --model_name_or_path "$MODEL_DIR" \
    --dataset_name "$DATA_FILE" \
    --output_dir "$OUTPUT_DIR" \
    --deepspeed_config "$DS_CONFIG" \
    --seed 42 \
    --per_device_train_batch_size "${BATCH_SIZE_PER_GPU}" \
    --learning_rate "${LEARNING_RATE}" \
    --num_train_epochs "${EPOCHS}" \
    --block_size "${BLOCK_SIZE}" \
    --num_workers "${NUM_WORKERS}" \
    --ckpt_strategy "${CKPT_STRATEGY}" \
    2>&1 | tee "$LOG_FILE"



