import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
from vllm import LLM, SamplingParams
from vllm.config import ObservabilityConfig

def test_single_completion_metrics():
    llm = LLM(
        model="microsoft/DialoGPT-small", 
        tensor_parallel_size=1, 
        gpu_memory_utilization=0.3,
        disable_log_stats=False,
        show_hidden_metrics_for_version="0.8.6",
        collect_detailed_traces="all",
    )
    prompt = "What is the capital of France?"
    params = SamplingParams(temperature=0.7, max_tokens=16, n=1)
    completions = llm.generate([prompt], params)
    c = completions[0]
    print("Completion object attributes:", dir(c))
    print("Completion object type:", type(c))
    m = getattr(c, "metrics", None)
    print("c.metrics:", m)
    if hasattr(c, 'outputs') and c.outputs:
        print("Output attributes:", dir(c.outputs[0]))
        output_metrics = getattr(c.outputs[0], "metrics", None)
        print("output metrics:", output_metrics)
    for attr_name in ['timing', 'stats', 'request_metrics', 'generation_metrics']:
        attr_val = getattr(c, attr_name, None)
        if attr_val:
            print(f"Found {attr_name}:", attr_val)
    if m is not None:
        print("ttft:", m.first_token_time - m.arrival_time)
        print("decode_time:", m.last_token_time - m.first_token_time)
        print("total_time:", m.finished_time - m.arrival_time)
        print("tokens_generated:", len(c.outputs[0].token_ids))
        print("tokens_per_second:", len(c.outputs[0].token_ids)/(m.finished_time - m.arrival_time))
        print("prompt_tokens:", len(c.prompt))
        print("completion_tokens:", len(c.outputs[0].token_ids))
        print("batch_total_time:", m.finished_time - m.arrival_time)
    else:
        print("No metrics found - trying alternative methods...")
        import time
        print("Manual timing would be needed")

def test_batch_metrics():
    llm = LLM(
        model="microsoft/DialoGPT-small", 
        tensor_parallel_size=1, 
        gpu_memory_utilization=0.3,
        disable_log_stats=False,
        show_hidden_metrics_for_version="0.8.6",
        collect_detailed_traces="all",
    )
    prompts = ["Q1?", "Q2?", "Q3?"]
    params = SamplingParams(temperature=0.7, max_tokens=8, n=1)
    completions = llm.generate(prompts, params)
    if completions[0].metrics is None:
        print("No metrics available in batch test either")
        return
    times = [c.metrics for c in completions]
    batch_start = min(m.arrival_time for m in times)
    batch_end = max(m.finished_time for m in times)
    print("batch_total_time:", batch_end - batch_start)
    for i, c in enumerate(completions):
        print(f"completion {i} ttft:", c.metrics.first_token_time - c.metrics.arrival_time)

if __name__ == "__main__":
    print("Single completion metrics:")
    test_single_completion_metrics()
    print("\nBatch metrics:")
    test_batch_metrics()
