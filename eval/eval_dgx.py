"""
DGX H100 optimized evaluation script with enhanced batching and NVML monitoring
Refactored for cloud/server deployment with focus on GPU performance
"""
import os
import sys
import time
import json
import argparse
import importlib.util
from math import comb
from datetime import datetime
from tqdm import tqdm

# VLLM and ML libraries
from transformers import AutoTokenizer
from vllm import LLM, SamplingParams

# Local utilities  
from utils.utils import set_seed, load_jsonl, save_jsonl, construct_prompt
from utils.parser import *
from utils.data_loader import load_data
from utils.math_normalization import *
from utils.grader import *
# DGX-optimized instrumentation
from instrument.nvml_telemetry import NVMLTelemetryHandler
from instrument.batched_performance import VLLMBatchedPerformanceInstrument, measure_batch_inference
from utils.model_config import ModelConfigHelper
from utils.dgx_summary import DGXSummary

def parse_list(arg):
    return arg.split(',')

def save_completions(completions, filepath):
    """Save VLLM completions to pickle file"""
    import pickle
    with open(filepath, 'wb') as file:
        pickle.dump(completions, file)

def parse_args():
    parser = argparse.ArgumentParser(description="DGX H100 optimized LLM evaluation with GPU monitoring")
    
    # Model and data arguments
    parser.add_argument('--model_name_or_path', type=str, required=True, help="Model path or HuggingFace model name")
    parser.add_argument('--data_dir', default="./data", type=str, help="Data directory")
    parser.add_argument('--data_name', type=str, default="aime", help="Dataset name")
    parser.add_argument("--split", default="test2024", type=str, help="Data split")
    
    # Inference parameters
    parser.add_argument('--n_sampling', type=int, default=1, help="Number of samples per question")
    parser.add_argument("--k", type=int, default=1, help="Value of k for pass@k calculation")
    parser.add_argument("--temperature", default=0.0, type=float, help="Sampling temperature")
    parser.add_argument("--max_tokens", default=4096, type=int, help="Maximum tokens to generate")
    parser.add_argument("--top_p", default=1.0, type=float, help="Top-p sampling")
    parser.add_argument('--stop', type=parse_list, help="Stop sequences")
    
    # Batching parameters (optimized for H100)
    parser.add_argument("--batch_size", default=32, type=int, help="Batch size for inference")
    parser.add_argument("--max_batch_size", default=64, type=int, help="Maximum batch size")
    
    # Prompt and formatting
    parser.add_argument("--prompt_type", default="qwen-instruct", type=str, help="Prompt template type")
    parser.add_argument("--prompt_file_path", default="./prompts", type=str, help="Prompt templates directory")
    parser.add_argument("--surround_with_messages", action="store_true", help="Use chat template")
    parser.add_argument("--use_few_shot", action="store_true", help="Use few-shot examples")
    
    # Data range
    parser.add_argument('--start_idx', type=int, default=0, help="Start index for data slice")
    parser.add_argument('--end_idx', type=int, default=-1, help="End index for data slice (-1 for all)")
    
    # Output and logging
    parser.add_argument("--output_dir", default="./outputs", type=str, help="Output directory")
    parser.add_argument("--completions_save_dir", default='./completions', type=str, help="Completions save directory")
    parser.add_argument("--config_suffix", default="", type=str, help="Configuration suffix for file naming")
    
    # GPU and system configuration
    parser.add_argument("--gpu_memory_utilization", default=0.95, type=float, help="GPU memory utilization ratio")
    parser.add_argument("--tensor_parallel_size", default=None, type=int, help="Tensor parallel size (auto-detect if None)")
    parser.add_argument("--dtype", default='auto', type=str, help="Model dtype")
    
    # Monitoring configuration
    parser.add_argument("--telemetry_interval", default=0.5, type=float, help="GPU telemetry sampling interval (seconds)")
    parser.add_argument("--disable_telemetry", action="store_true", help="Disable GPU telemetry")
    
    # Other
    parser.add_argument("--seed", default=42, type=int, help="Random seed")
    
    args = parser.parse_args()
    
    # Auto-adjust parameters
    args.top_p = 1.0 if args.temperature == 0 else args.top_p
    
    if args.tensor_parallel_size is None:
        # Auto-detect tensor parallel size using model config helper
        available_gpus = os.environ.get('CUDA_VISIBLE_DEVICES', '0').split(',')
        num_gpus = len(available_gpus)
        
        if num_gpus == 1:
            args.tensor_parallel_size = 1
            print(f"Single GPU detected - using tensor_parallel_size=1 for optimal performance")
        else:
            model_helper = ModelConfigHelper(args.model_name_or_path)
            args.tensor_parallel_size = model_helper.get_optimal_tensor_parallel_size(num_gpus)
    
    print(f"Configuration:")
    print(f"  Model: {args.model_name_or_path}")
    print(f"  Dataset: {args.data_name} ({args.split})")
    print(f"  Batch size: {args.batch_size}")
    print(f"  Tensor parallel size: {args.tensor_parallel_size}")
    print(f"  GPU memory utilization: {args.gpu_memory_utilization}")
    print(f"  Stop sequences: {args.stop}")
    
    return args

def get_conversation_prompt_by_messages(tokenizer, messages):
    """Apply chat template to messages"""
    text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True
    )
    return text

def get_three_prompt(prompt_type, data_name):
    """Load prompt templates dynamically"""
    file_path = os.path.join(".", "prompts", prompt_type, f"{data_name}.py")
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Prompt file not found: {file_path}")
    
    spec = importlib.util.spec_from_file_location("dynamic_module", file_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    
    system_prompt = getattr(module, 'system_prompt', "You are a helpful assistant.")
    few_shot_prompt = getattr(module, 'few_shot_prompt', "")
    question_format = getattr(module, 'question_format', "Question: {question}\nAnswer:")
    
    return system_prompt, few_shot_prompt, question_format

def create_batches(items, batch_size):
    """Create batches from list of items"""
    for i in range(0, len(items), batch_size):
        yield items[i:i + batch_size]

def infer_dgx(args):
    """Main inference function optimized for DGX H100 systems"""
    print(f"=== DGX H100 Optimized Evaluation ===")
    print(f"Model: {args.model_name_or_path}")
    print(f"Target batch size: {args.batch_size}")
    
    # Load and prepare data
    examples = load_data(args.data_name, args.split, args.data_dir)
    if args.end_idx == -1:
        args.end_idx = len(examples)
    examples = examples[args.start_idx:args.end_idx]
    
    print(f"Processing {len(examples)} questions (indices {args.start_idx}:{args.end_idx})")
    
    # Set up output paths
    dt_string = datetime.now().strftime("%m-%d_%H-%M")
    model_name = args.model_name_or_path.split("/")[-1]
    out_file_prefix = f'{args.max_tokens}t_{args.split}_{args.prompt_type}_t{args.temperature}_b{args.batch_size}'
    out_file = f'{args.output_dir}/{model_name}/{args.data_name}/{out_file_prefix}_k{args.n_sampling}_s{args.start_idx}_e{args.end_idx}.jsonl'
    
    # Check if output already exists
    if os.path.exists(out_file):
        print(f"Output file already exists: {out_file}")
        print("Skipping generation to avoid overwrite")
        return
    
    # Create output directories
    os.makedirs(f'{args.output_dir}/{model_name}/{args.data_name}', exist_ok=True)
    os.makedirs(f'{args.completions_save_dir}/{model_name}/{args.data_name}', exist_ok=True)
    
    # Initialize instrumentation
    config_suffix = args.config_suffix or f"dgx_batch{args.batch_size}_tokens{args.max_tokens}"
    
    summary = DGXSummary(args, model_name)
    perf_instrument = VLLMBatchedPerformanceInstrument(
        output_dir=f'{args.output_dir}/{model_name}/{args.data_name}',
        model_name=model_name,
        config_suffix=config_suffix
    )
    
    # Initialize tokenizer and prepare prompts
    print("Loading tokenizer and preparing prompts...")
    tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path, trust_remote_code=True)
    
    system_prompt, few_shot_prompt, question_format = get_three_prompt(args.prompt_type, args.data_name)
    
    prompt_batch = []
    for example in tqdm(examples, desc="Preparing prompts"):
        question = parse_question(example, args.data_name)
        
        if args.use_few_shot:
            cur_prompt = few_shot_prompt + question_format.format(question=question)
        else:
            cur_prompt = question_format.format(question=question)
            
        if args.surround_with_messages:
            messages = [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": cur_prompt}
            ]
            cur_prompt = get_conversation_prompt_by_messages(tokenizer=tokenizer, messages=messages)
        
        prompt_batch.append(cur_prompt)
    
    print(f"Sample prompt (first 500 chars):\n{prompt_batch[0][:500]}...")
    
    # Initialize VLLM model
    print("Initializing VLLM model...")
    llm = LLM(
        model=args.model_name_or_path,
        tensor_parallel_size=args.tensor_parallel_size,
        trust_remote_code=True,
        gpu_memory_utilization=args.gpu_memory_utilization,
        dtype=args.dtype,
        max_num_batched_tokens=args.max_batch_size * args.max_tokens,
        max_num_seqs=args.max_batch_size
    )
    
    # Set up sampling parameters
    sampling_params = SamplingParams(
        temperature=args.temperature,
        max_tokens=args.max_tokens,
        n=args.n_sampling,
        top_p=args.top_p,
        stop=args.stop,
        seed=args.seed,
    )
    
    print(f"Sampling parameters: temp={args.temperature}, max_tokens={args.max_tokens}, n={args.n_sampling}")
    
    # Start GPU telemetry monitoring
    telemetry_handler = None
    if not args.disable_telemetry:
        try:
            telemetry_handler = NVMLTelemetryHandler(
                output_dir=f'{args.output_dir}/{model_name}/{args.data_name}',
                config_suffix=config_suffix,
                sampling_interval=args.telemetry_interval
            )
            telemetry_handler.start_logging()
            print("GPU telemetry monitoring started")
        except Exception as e:
            print(f"Warning: Could not start GPU telemetry: {e}")
            telemetry_handler = None
    
    try:
        # Batch processing
        print(f"Starting batched inference with batch size {args.batch_size}...")
        
        all_completions = []
        file_outputs = []
        
        # Create batches of prompts
        prompt_batches = list(create_batches(prompt_batch, args.batch_size))
        example_batches = list(create_batches(examples, args.batch_size))
        
        print(f"Processing {len(prompt_batches)} batches")
        
        for batch_idx, (batch_prompts, batch_examples) in enumerate(tqdm(
            zip(prompt_batches, example_batches), 
            desc="Processing batches", 
            total=len(prompt_batches)
        )):
            
            # Generate completions for this batch
            with measure_batch_inference() as batch_timer:
                batch_completions = llm.generate(batch_prompts, sampling_params)
                batch_timer.record_completion(batch_completions)
            
            # Record performance metrics
            question_ids = list(range(batch_idx * args.batch_size, batch_idx * args.batch_size + len(batch_examples)))
            perf_instrument.record_batch_inference(
                batch_id=batch_idx,
                completions=batch_completions,
                batch_start_time=batch_timer.start_time,
                batch_end_time=batch_timer.end_time,
                question_ids=question_ids
            )
            
            # Save completions for this batch
            completions_save_file = f'{args.completions_save_dir}/{model_name}/{args.data_name}/{out_file_prefix}_batch{batch_idx}.pkl'
            save_completions(batch_completions, completions_save_file)
            
            # Process results for this batch
            for i, (example, completion) in enumerate(zip(batch_examples, batch_completions)):
                question = parse_question(example, args.data_name)
                generated_responses = [output.text for output in completion.outputs]
                
                result_dict = {
                    "question": question,
                    "generated_responses": generated_responses,
                }
                
                # Preserve original metadata
                if "id" in example:
                    result_dict["id"] = example["id"]
                if "source" in example:
                    result_dict["source"] = example["source"]
                
                file_outputs.append(result_dict)
            
            all_completions.extend(batch_completions)
            
            # Print batch progress
            if batch_idx % 10 == 0 or batch_idx == len(prompt_batches) - 1:
                batch_throughput = len(batch_prompts) / batch_timer.duration
                print(f"  Batch {batch_idx + 1}/{len(prompt_batches)}: "
                      f"{len(batch_prompts)} questions in {batch_timer.duration:.2f}s "
                      f"({batch_throughput:.1f} questions/s)")
        
        print("Inference completed. Processing results...")
        
    finally:
        # Stop telemetry monitoring and collect stats
        if telemetry_handler:
            telemetry_handler.stop_logging()
            gpu_stats = telemetry_handler.get_summary_stats()
            summary.add_gpu_telemetry(gpu_stats)
    
    # Save performance metrics and add to summary
    perf_instrument.save_metrics()
    perf_stats = perf_instrument.get_summary_stats()
    summary.add_performance_metrics(perf_stats)
    
    # Print summaries using modular approach
    summary.print_all_summaries()
    
    # Evaluate correctness
    print("Evaluating answer correctness...")
    correct_cnt = 0
    pass_at_k_list = []
    
    for i, (example, result) in enumerate(tqdm(zip(examples, file_outputs), desc="Checking correctness")):
        gt_cot, gt_ans = parse_ground_truth(example, args.data_name)
        generated_responses = result['generated_responses']
        generated_answers = [extract_answer(response, args.data_name) for response in generated_responses]
        is_correct_list = [check_is_correct(answer, gt_ans) for answer in generated_answers]
        is_correct = any(is_correct_list)
        
        if is_correct:
            correct_cnt += 1
        
        # Add evaluation results to output
        file_outputs[i]['generated_answers'] = generated_answers
        file_outputs[i]['gold_answer'] = gt_ans
        file_outputs[i]['is_correct'] = is_correct
        file_outputs[i]['answers_correctness'] = is_correct_list
        
        # Calculate pass@k if multiple samples
        if len(is_correct_list) > 1:
            correct_answers = sum(is_correct_list)
            n = len(generated_answers)
            if correct_answers > 0:
                if n - correct_answers < args.k:
                    pass_at_k = 1.0
                else:
                    pass_at_k = 1 - (comb(n - correct_answers, args.k) / comb(n, args.k))
                pass_at_k_list.append(pass_at_k)
            else:
                pass_at_k_list.append(0.0)
    
    # Save results
    print("Saving results...")
    temp_out_file = out_file + ".tmp"
    with open(temp_out_file, 'w', encoding='utf-8') as f:
        for result in tqdm(file_outputs, desc="Writing results"):
            f.write(json.dumps(result, ensure_ascii=False))
            f.write("\n")
    os.rename(temp_out_file, out_file)
    
    # Print final results
    accuracy = correct_cnt / len(examples)
    print(f"\nFinal Results:")
    print(f"  Correct: {correct_cnt}/{len(examples)}")
    print(f"  Accuracy: {accuracy:.4f}")
    
    if pass_at_k_list:
        average_pass_at_k = sum(pass_at_k_list) / len(pass_at_k_list)
        print(f"  Pass@{args.k}: {average_pass_at_k:.4f}")
        summary.add_results(len(examples), correct_cnt, average_pass_at_k)
    else:
        print(f"  Pass@1: {accuracy:.4f}")
        summary.add_results(len(examples), correct_cnt)
    
    # Save summary
    summary.save(out_file)
    print(f"Results saved to: {out_file}")

if __name__ == "__main__":
    args = parse_args()
    set_seed(args.seed)
    infer_dgx(args)
