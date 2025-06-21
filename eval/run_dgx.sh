#!/bin/bash

# DGX H100 optimized evaluation runner
# Usage: ./run_dgx.sh [model_path] [batch_size] [max_tokens]

set -e

MODEL_PATH=${1:-"l3lab/L1-Qwen-1.5B-Max"}
BATCH_SIZE=${2:-1}
MAX_TOKENS=${3:-32768}

echo "=== DGX H100 LLM Evaluation ==="
echo "Model: $MODEL_PATH"
echo "Batch Size: $BATCH_SIZE"
echo "Max Tokens: $MAX_TOKENS"
echo ""

# Check CUDA availability
if ! command -v nvidia-smi &> /dev/null; then
    echo "Error: nvidia-smi not found. Please ensure CUDA is available."
    exit 1
fi

# Display GPU information
echo "Available GPUs:"
nvidia-smi --list-gpus
echo ""

# Set default CUDA devices if not specified
if [ -z "$CUDA_VISIBLE_DEVICES" ]; then
    export CUDA_VISIBLE_DEVICES="1"
fi

echo "Using GPUs: $CUDA_VISIBLE_DEVICES"
echo ""

# Set environment variables to reduce warnings
export TOKENIZERS_PARALLELISM=false
export OMP_NUM_THREADS=1

# Check if pynvml is installed
python -c "import pynvml" 2>/dev/null || {
    echo "Installing pynvml for GPU monitoring..."
    pip install pynvml
}

# Run the evaluation
echo "Starting evaluation..."
python eval_dgx.py \
    --model_name_or_path "$MODEL_PATH" \
    --data_name "aime" \
    --prompt_type "qwen-instruct" \
    --temperature 0.6 \
    --max_tokens $MAX_TOKENS \
    --batch_size $BATCH_SIZE \
    --max_batch_size $((BATCH_SIZE * 2)) \
    --n_sampling 1 \
    --k 1 \
    --split "test2024" \
    --seed 42 \
    --top_p 0.95 \
    --surround_with_messages \
    --output_dir "./outputs" \
    --completions_save_dir "./completions" \
    --gpu_memory_utilization 0.85 \
    --telemetry_interval 0.5 \
    --config_suffix "b${BATCH_SIZE}_t${MAX_TOKENS}"

echo ""
echo "DGX evaluation completed!"
echo "Results saved in: ./outputs/"
echo "GPU telemetry saved with performance metrics"
