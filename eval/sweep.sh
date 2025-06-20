#!/bin/bash

# Parameter sweep script for max_tokens evaluation
# Usage: ./sweep.sh [output_base_dir] [model_path]

OUTPUT_BASE_DIR=${1:-"./sweep_results"}
MODEL_PATH=${2:-"agentica-org/DeepScaleR-1.5B-Preview"}

echo "Starting parameter sweep..."
echo "Output directory: $OUTPUT_BASE_DIR"
echo "Model: $MODEL_PATH"

CUDA_VISIBLE_DEVICES='0' \
python instrument/sweep.py \
--model_name_or_path "$MODEL_PATH" \
--data_name "aime" \
--prompt_type "qwen-instruct" \
--temperature 0.6 \
--n_sampling 1 \
--k 1 \
--split "test2024" \
--seed 0 \
--top_p 0.95 \
--surround_with_messages \
--output_base_dir "$OUTPUT_BASE_DIR" \
--token_values 8192

echo "Parameter sweep completed. Results saved to: $OUTPUT_BASE_DIR"