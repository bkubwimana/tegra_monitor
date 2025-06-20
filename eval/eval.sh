# export OMP_NUM_THREADS=1 
# export VLLM_WORKER_MULTIPROC_METHOD=spawn 
# CUDA_LAUNCH_BLOCKING=1
# TORCH_USE_CUDA_DSA=1

CUDA_VISIBLE_DEVICES='0,1' \
python eval.py \
--model_name_or_path "agentica-org/DeepScaleR-1.5B-Preview" \
--data_name "aime" \
--prompt_type "qwen-instruct" \
--temperature 0.6 \
--start_idx 0 \
--end_idx -1 \
--n_sampling 1 \
--k 1 \
--split "test2024" \
--max_tokens 32768 \
--seed 0 \
--top_p 0.95 \
--surround_with_messages 

