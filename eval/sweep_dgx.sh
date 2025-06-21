#!/bin/bash

# DGX H100 parameter sweep runner
# Usage: ./sweep_dgx.sh [model_path] [sweep_type]

set -e

MODEL_PATH=${1:-"agentica-org/DeepScaleR-1.5B-Preview"}
SWEEP_TYPE=${2:-"batch"} 

echo "=== DGX H100 Parameter Sweep ==="
echo "Model: $MODEL_PATH"
echo "Sweep Type: $SWEEP_TYPE"
echo ""

# Set CUDA devices (use all available H100s)
if [ -z "$CUDA_VISIBLE_DEVICES" ]; then
    export CUDA_VISIBLE_DEVICES="0"
fi

echo "Using GPUs: $CUDA_VISIBLE_DEVICES"

# Check dependencies
python -c "import pynvml" 2>/dev/null || {
    echo "Installing pynvml for GPU monitoring..."
    pip install pynvml
}

# Create output directory with timestamp
TIMESTAMP=$(date +"%m%d_%H%M")
OUTPUT_DIR="./sweep_results_${TIMESTAMP}"

echo "Output directory: $OUTPUT_DIR"
echo ""

# Define sweep parameters based on type
case $SWEEP_TYPE in
    "batch")
        echo "Running batch size sweep: [1,5,10,15]"
        python instrument/dgx_sweep.py \
            --model_name_or_path "$MODEL_PATH" \
            --output_base_dir "$OUTPUT_DIR" \
            --sweep_type "batch" \
            --batch_sizes 1 5 10 15 \
            --data_name "aime" \
            --prompt_type "qwen-instruct" \
            --temperature 0.6 \
            --max_tokens 32768 \
            --surround_with_messages
        ;;
    "tokens")
        echo "Running token sweep: [8192, 16384]"
        python instrument/dgx_sweep.py \
            --model_name_or_path "$MODEL_PATH" \
            --output_base_dir "$OUTPUT_DIR" \
            --sweep_type "tokens" \
            --batch_sizes 5 \
            --token_values 8192 16384 \
            --data_name "aime" \
            --prompt_type "qwen-instruct" \
            --temperature 0.6 \
            --surround_with_messages
        ;;
    "combined")
        echo "Running combined sweep: batch=[1 5 10 15] x tokens=[8192,16384,32768]"
        python instrument/dgx_sweep.py \
            --model_name_or_path "$MODEL_PATH" \
            --output_base_dir "$OUTPUT_DIR" \
            --sweep_type "combined" \
            --batch_sizes 1 5 10 15 \
            --token_values 8192 16384 32768 \
            --data_name "aime" \
            --prompt_type "qwen-instruct" \
            --temperature 0.6 \
            --surround_with_messages
        ;;
    *)
        echo "Error: Unknown sweep type '$SWEEP_TYPE'"
        echo "Valid types: batch, tokens, combined"
        exit 1
        ;;
esac

echo ""
echo "Parameter sweep completed!"
echo "Results saved in: $OUTPUT_DIR"
echo ""
echo "Summary files:"
echo "  - dgx_sweep_summary.json (detailed results)"  
echo "  - dgx_sweep_results.csv (analysis-ready data)"
echo ""
echo "To analyze results, you can:"
echo "  cat $OUTPUT_DIR/dgx_sweep_summary.json | jq '.sweep_info'"
echo "  python -c \"import pandas as pd; print(pd.read_csv('$OUTPUT_DIR/dgx_sweep_results.csv'))\""
