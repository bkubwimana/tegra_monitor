# LLM Evaluation Suite 

LLM MATH evaluation system optimized for cerver class NVIDIA GPUs with comprehensive GPU monitoring, batched inference, and automated parameter sweeping.

## 🚀 Key Features

- **H100-Optimized Batching**: Efficient batch processing with configurable batch sizes (8-128+)
- **NVML GPU Monitoring**: Real-time GPU telemetry using NVIDIA Management Library
- **Multi-GPU Support**: Automatic tensor parallelism across multiple H100 GPUs  
- **Advanced Performance Metrics**: Throughput, token generation rates, memory utilization
- **Parameter Sweeping**: Automated exploration of batch sizes and token limits
- **Production-Ready**: Robust error handling, comprehensive logging, CSV outputs

## 🏗️ Architecture

```
eval/
├── eval_dgx.py                 # Main DGX-optimized evaluation script
├── instrument/
│   ├── nvml_telemetry.py      # NVML-based GPU monitoring
│   ├── batched_performance.py  # Batched inference performance tracking
│   └── dgx_sweep.py           # Parameter sweep automation
├── run_dgx.sh                 # Single evaluation runner
├── sweep_dgx.sh               # Parameter sweep runner
└── requirements_dgx.txt       # DGX-specific dependencies
```

## 📋 Prerequisites

### Hardware
- NVIDIA DGX system with H100 GPUs
- CUDA 12.0+ recommended
- Sufficient GPU memory for your target model

### Software
- Python 3.8+
- NVIDIA drivers with NVML support
- VLLM library for efficient inference

## 🛠️ Installation

```bash
# Navigate to eval directory
cd /path/to/tegra_monitor/eval

# Install DGX-specific requirements
pip install -r requirements_dgx.txt

# Make scripts executable
chmod +x *.sh

# Verify GPU access
nvidia-smi
python -c "import pynvml; pynvml.nvmlInit(); print('NVML initialized successfully')"
```

## 🚀 Quick Start

### Single Evaluation

```bash
# Basic evaluation with default settings
./run_dgx.sh "meta-llama/Llama-2-7b-hf"

# Custom batch size and token limit
./run_dgx.sh "meta-llama/Llama-2-7b-hf" 64 8192

# Advanced usage
python eval_dgx.py \
    --model_name_or_path "meta-llama/Llama-2-7b-hf" \
    --batch_size 32 \
    --max_tokens 4096 \
    --data_name "aime" \
    --prompt_type "qwen-instruct" \
    --gpu_memory_utilization 0.95 \
    --tensor_parallel_size 8
```

### Parameter Sweeps

```bash
# Batch size sweep
./sweep_dgx.sh "meta-llama/Llama-2-7b-hf" batch

# Token length sweep  
./sweep_dgx.sh "meta-llama/Llama-2-7b-hf" tokens

# Combined parameter sweep
./sweep_dgx.sh "meta-llama/Llama-2-7b-hf" combined
```

## 📊 Output Files

### Performance Metrics
- `performance_*.csv`: Batch-level performance metrics
- `performance_detailed_*.csv`: Per-question timing breakdown

### GPU Telemetry  
- `gpu_telemetry_*.csv`: Comprehensive GPU monitoring data

### Evaluation Results
- `*.jsonl`: Standard evaluation results with answers and correctness

### Sweep Summaries
- `dgx_sweep_summary.json`: Detailed sweep results
- `dgx_sweep_results.csv`: Analysis-ready tabular data

## ⚙️ Configuration Options

### Core Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `--batch_size` | 32 | Inference batch size |
| `--max_tokens` | 4096 | Maximum tokens to generate |
| `--gpu_memory_utilization` | 0.95 | GPU memory usage ratio |
| `--tensor_parallel_size` | auto | Number of GPUs for tensor parallelism |
| `--telemetry_interval` | 0.5 | GPU monitoring sample rate (seconds) |

### Model Configuration

```bash
python eval_dgx.py \
    --model_name_or_path "microsoft/DialoGPT-large" \
    --batch_size 64 \
    --max_batch_size 128 \
    --gpu_memory_utilization 0.90 \
    --tensor_parallel_size 4 \
    --dtype "float16"
```

### Dataset Options

```bash
# AIME mathematical reasoning
--data_name "aime" --split "test2024"

# MATH dataset  
--data_name "math" --split "test"

# Custom dataset (extend data_loader.py)
--data_name "custom" --data_dir "./my_data"
```

## 📈 GPU Monitoring

The NVML telemetry system captures 20+ GPU metrics:

### Power & Temperature
- Power draw (W) and power limits
- GPU temperature and thermal throttling
- Fan speeds and performance states

### Memory & Compute
- GPU and memory utilization percentages
- Memory usage (used/total MB)
- SM and memory clock frequencies

### Interconnect
- PCIe generation, width, and throughput
- NVLink utilization (H100-specific)
- Inter-GPU communication bandwidth

### Example GPU Metrics CSV
```csv
timestamp,gpu_id,temperature_c,power_draw_w,memory_util_pct,gpu_util_pct,sm_clock_mhz
1703123456.789,0,65.2,380.5,87.3,95.1,1980
1703123457.289,0,66.1,385.2,89.1,94.8,1980
```

## 🔧 Performance Optimization

### Batch Size Selection
- **Small models (7B)**: Start with batch_size=64
- **Large models (70B)**: Start with batch_size=16-32  
- **Memory constrained**: Reduce batch_size, increase max_batch_size

### Multi-GPU Scaling
```bash
# Use all 8 H100s
export CUDA_VISIBLE_DEVICES="0,1,2,3,4,5,6,7"
--tensor_parallel_size 8

# Use subset of GPUs
export CUDA_VISIBLE_DEVICES="0,1,2,3"  
--tensor_parallel_size 4
```

### Memory Management
```bash
# Conservative memory usage
--gpu_memory_utilization 0.85

# Aggressive memory usage (monitor for OOM)
--gpu_memory_utilization 0.98
```

## 📊 Analysis Examples

### Performance Analysis
```python
import pandas as pd
import matplotlib.pyplot as plt

# Load performance data
df = pd.read_csv('performance_dgx_batch32_tokens4096.csv')

# Plot throughput over time
plt.figure(figsize=(12, 6))
plt.plot(df['timestamp'], df['avg_tokens_per_second'])
plt.xlabel('Time')
plt.ylabel('Tokens/Second')
plt.title('Token Generation Throughput')
plt.show()

# Summary statistics
print(f"Average throughput: {df['avg_tokens_per_second'].mean():.2f} tokens/s")
print(f"Peak throughput: {df['avg_tokens_per_second'].max():.2f} tokens/s")
```

### GPU Utilization Analysis
```python
# Load GPU telemetry
gpu_df = pd.read_csv('gpu_telemetry_dgx_batch32_tokens4096.csv')

# Per-GPU summary
for gpu_id in gpu_df['gpu_id'].unique():
    gpu_data = gpu_df[gpu_df['gpu_id'] == gpu_id]
    print(f"GPU {gpu_id}:")
    print(f"  Avg Utilization: {gpu_data['gpu_util_pct'].mean():.1f}%")
    print(f"  Avg Power: {gpu_data['power_draw_w'].mean():.1f}W")
    print(f"  Peak Temperature: {gpu_data['temperature_c'].max():.1f}°C")
```

### Energy Consumption
```python
# Calculate total energy consumption
total_energy_j = 0
for gpu_id in gpu_df['gpu_id'].unique():
    gpu_data = gpu_df[gpu_df['gpu_id'] == gpu_id]
    # Approximate energy as power × time
    avg_power_w = gpu_data['power_draw_w'].mean()
    duration_s = (gpu_data['timestamp'].max() - gpu_data['timestamp'].min())
    energy_j = avg_power_w * duration_s
    total_energy_j += energy_j

print(f"Total energy consumption: {total_energy_j:.1f} J ({total_energy_j/3600:.3f} Wh)")
```

## 🔍 Parameter Sweep Analysis

```python
# Load sweep results
sweep_df = pd.read_csv('dgx_sweep_results.csv')

# Find optimal configurations
best_throughput = sweep_df.loc[sweep_df['questions_per_second'].idxmax()]
best_efficiency = sweep_df.assign(
    efficiency=sweep_df['questions_per_second'] / sweep_df['avg_power_w']
).loc[sweep_df.assign(
    efficiency=sweep_df['questions_per_second'] / sweep_df['avg_power_w']
)['efficiency'].idxmax()]

print(f"Best throughput: batch={best_throughput['batch_size']}, "
      f"tokens={best_throughput['max_tokens']}, "
      f"{best_throughput['questions_per_second']:.2f} q/s")

print(f"Most efficient: batch={best_efficiency['batch_size']}, "
      f"tokens={best_efficiency['max_tokens']}, "
      f"{best_efficiency['efficiency']:.3f} q/s/W")
```

## 🚨 Troubleshooting

### Common Issues

**GPU Memory Errors**
```bash
# Reduce memory utilization
--gpu_memory_utilization 0.80

# Reduce batch size
--batch_size 16
```

**NVML Initialization Failed**
```bash
# Check NVIDIA drivers
nvidia-smi

# Reinstall pynvml
pip uninstall pynvml nvidia-ml-py3
pip install pynvml
```

**Low GPU Utilization**
```bash
# Increase batch size
--batch_size 64

# Check tensor parallelism
--tensor_parallel_size 8
```

**Performance Bottlenecks**
```bash
# Enable tensor parallelism
export CUDA_VISIBLE_DEVICES="0,1,2,3,4,5,6,7"

# Optimize memory usage
--max_batch_size 128

# Use appropriate dtype
--dtype "float16"  # or "bfloat16" for H100
```

## 🔬 Advanced Usage

### Custom Datasets
1. Add dataset loader to `utils/data_loader.py`
2. Add prompt template to `prompts/[prompt_type]/[dataset].py`  
3. Add answer extraction to `utils/parser.py`

### Custom Metrics
1. Extend `GPUMetrics` class in `nvml_telemetry.py`
2. Add metric collection in `_collect_gpu_metrics()`
3. Update CSV headers and row formatting

### Integration with MLflow
```python
import mlflow

# Log sweep results to MLflow
for result in sweep_results:
    with mlflow.start_run():
        mlflow.log_params({
            'batch_size': result['batch_size'],
            'max_tokens': result['max_tokens']
        })
        mlflow.log_metrics(result['metrics'])
```

## 📄 License

This project maintains the same license as the parent tegra_monitor repository.

## 🤝 Contributing

1. Focus on DGX/cloud optimizations (avoid edge device code)
2. Maintain compatibility with VLLM updates
3. Add comprehensive GPU monitoring for new architectures
4. Include performance benchmarks in pull requests

---

**Performance Note**: This system is optimized for NVIDIA DGX H100 systems. Performance on other hardware may vary. For edge deployment, use the original Tegra Orin codebase.



## REAMDE FOR Normal eval with no performance monitoring


# Math Problem Evaluation Framework

This repository contains scripts for evaluating Large Language Models (LLMs) on mathematical reasoning tasks. The evaluation framework includes both inference (using the VLLM framework) and evaluation (using both rule-based and model-based approaches) components.

## Environment Setup

When setting up the environment, pay attention to package version numbers, especially for those with specific version requirements noted in the documentation.

```bash
pip install -r requirements.txt
```

## Benchmark Evaluation

### Data Preparation

All benchmark datasets for evaluation should be placed in the `./data` directory.

To add a new test dataset, follow the format of existing benchmarks in the `./data` directory.

### Prompt Configuration

For mathematical problems, we use the Qwen-instruct template:

```python
system_prompt = "Please reason step by step, and put your final answer within \\boxed{}."

few_shot_prompt = ""

question_format = """{question}"""
```

When adding a new mathematics benchmark, you can directly copy the above content to the corresponding `./prompts/qwen-instruct/xxx.py` file.

### Rule-Based Evaluation Interface

The framework provides a simple interface for rule-based evaluation of model predictions. Here's a basic example of how to use it:

```python
from utils.grader import check_is_correct
from utils.parser import extract_answer

def evaluate_prediction(model_pred: str, gold_answer: str) -> bool:
    """
    Evaluate a model's prediction against a gold answer.
    
    Args:
        model_pred (str): The model's prediction with answer in \boxed{}.
        gold_answer (str): The correct answer to compare against.
    
    Returns:
        bool: True if the prediction matches the gold answer, False otherwise.
    """
    # Extract the answer from model prediction
    extracted_answer = extract_answer(model_pred)
    
    # Check if the extracted answer matches the gold answer
    return check_is_correct(extracted_answer, gold_answer)

if __name__ == "__main__":
    # Example usage
    model_pred = "Let's solve this step by step:\n1. First...\n2. Then...\nSo the final answer is \\boxed{\\frac{1}{4}}"
    gold_answer = "0.25"
    
    is_correct = evaluate_prediction(model_pred, gold_answer)
    print(f"Prediction is correct: {is_correct}")  # True
```

The evaluation utilities handle various answer formats:
- Fractions (e.g., "\\frac{1}{4}")
- Decimals (e.g., "0.25")
- Mixed numbers
- Mathematical expressions

### Running Evaluation

Execute the evaluation script using:

```bash
bash eval.sh
```

Parameters in `eval.sh`:

```bash
CUDA_VISIBLE_DEVICES='0,1,2,3' \
python eval.py \
--model_name_or_path "/path/to/model/weights" \  # Path to model weights
--data_name "math" \  # Benchmark name (corresponding to first-level directory in ./data)
--prompt_type "qwen-instruct" \  # Default chat template
--temperature 0.0 \  # Sampling temperature
--start_idx 0 \  # Starting index for evaluation data
--end_idx -1 \  # Ending index for evaluation data
--n_sampling 1 \  # Number of samples per question
--k 1 \  # k value for unbiased pass@k calculation
--split "test" \  # Benchmark subset partition
--max_tokens 32768 \  # Maximum output length
--seed 0 \  # Random seed
--top_p 1 \  # Top-p sampling parameter
--surround_with_messages \  # Enable this flag if using chat template
```

## Model-Based Evaluation

While rule-based evaluation works well for structured answers (e.g., multiple choice questions, pure numerical responses) like those in AIME and most MATH problems, more complex response types (expressions, equations, or simple natural language descriptions) require model-based evaluation.

We use Qwen2.5-32B-Instruct as our judge model due to its excellent instruction-following capabilities and strong foundational knowledge. For reference, our evaluation prompts can be found in [`prompt.txt`](https://github.com/GAIR-NLP/LIMO/blob/main/eval/prompt.txt).


## Acknowledgments

Our evaluation code is modified from [Qwen2.5-Math](https://github.com/QwenLM/Qwen2.5-Math/tree/main/evaluation). We thank their team for their valuable contributions to the community.