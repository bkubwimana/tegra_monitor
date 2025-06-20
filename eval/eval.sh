export OMP_NUM_THREADS=1 
export VLLM_WORKER_MULTIPROC_METHOD=spawn 
CUDA_LAUNCH_BLOCKING=1
TORCH_USE_CUDA_DSA=1

CUDA_VISIBLE_DEVICES='0,1' \
python eval.py \
--model_name_or_path "deepseek-ai/DeepSeek-R1-Distill-Qwen-14B" \
--data_name "aime" \
--prompt_type "qwen-instruct" \
--temperature 0.0 \
--start_idx 0 \
--end_idx -1 \
--n_sampling 1 \
--k 1 \
--split "test2024" \
--max_tokens 24576 \
--seed 0 \
--top_p 1 \
--surround_with_messages 

