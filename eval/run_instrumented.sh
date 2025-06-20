#!/bin/bash

# Example usage of the instrumented evaluation system

echo "=== Instrumented Evaluation System ==="
echo "This system provides comprehensive performance monitoring and parameter sweeping"
echo ""

echo "Available options:"
echo "1. Run single instrumented evaluation"
echo "2. Run parameter sweep across max_tokens"
echo "3. Process existing telemetry logs"
echo ""

read -p "Select option (1-3): " choice

case $choice in
    1)
        echo "Running single instrumented evaluation..."
        CUDA_VISIBLE_DEVICES='0' python eval_instrumented.py \
            --model_name_or_path "agentica-org/DeepScaleR-1.5B-Preview" \
            --data_name "aime" \
            --prompt_type "qwen-instruct" \
            --temperature 0.6 \
            --max_tokens 2048 \
            --n_sampling 1 \
            --k 1 \
            --split "test2024" \
            --seed 0 \
            --top_p 0.95 \
            --surround_with_messages \
            --output_dir "./instrumented_outputs" \
            --config_suffix "single_run"
        ;;
    2)
        echo "Running parameter sweep..."
        ./sweep.sh "./sweep_results" "agentica-org/DeepScaleR-1.5B-Preview"
        ;;
    3)
        echo "Processing telemetry logs..."
        read -p "Enter log file path: " log_path
        read -p "Enter output directory: " output_dir
        python instrument/telemetry_proc.py --log_file "$log_path" --output_dir "$output_dir"
        ;;
    *)
        echo "Invalid option"
        exit 1
        ;;
esac

echo "Operation completed!"
