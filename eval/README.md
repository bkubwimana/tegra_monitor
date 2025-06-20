# LLM Evaluation with Performance & Energy Instrumentation

A comprehensive system for evaluating Large Language Models (LLMs) with detailed performance monitoring and energy consumption analysis on NVIDIA Jetson platforms.

## 🚀 Features

- **VLLM-Native Performance Monitoring**: Accurate TTFT, decode time, tokens/sec measurement
- **Comprehensive Energy Tracking**: Detailed tegrastats telemetry with 50+ metrics
- **Parameter Sweeping**: Automated testing across different configurations (max_tokens, etc.)
- **Professional CSV Output**: Wide-format data for analysis and visualization
- **Robust Error Handling**: Production-ready with graceful error recovery
- **Modular Design**: Clean separation of concerns for easy extension

## 📁 Project Structure

```
eval/
├── eval_instrumented.py       # Main instrumented evaluation script
├── eval.py                   # Original evaluation script (preserved)
├── instrument/               # Instrumentation modules
│   ├── performance.py        # VLLM performance timing
│   ├── telemetry.py         # Tegrastats logging management
│   ├── telemetry_proc.py    # Comprehensive telemetry parsing
│   └── sweep.py             # Parameter sweep automation
├── sweep.sh                 # Shell script for parameter sweeps
├── run_instrumented.sh      # Interactive script for common tasks
└── README.md               # This file
```

## 🛠️ Installation & Setup

### Prerequisites

- NVIDIA Jetson platform with tegrastats
- Python 3.8+
- VLLM library
- Required Python packages: `transformers`, `tqdm`, `pandas` (optional)

### Setup

```bash
# Navigate to the eval directory
cd /path/to/tegra_monitor/eval

# Ensure scripts are executable
chmod +x *.sh

# Install dependencies (if needed)
pip install vllm transformers tqdm
```

## 📊 Usage

### Option 1: Single Instrumented Evaluation

Run a single evaluation with comprehensive monitoring:

```bash
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
```

### Option 2: Parameter Sweep

Run automated parameter sweep across different max_tokens values:

```bash
# Using the convenience script
./sweep.sh "./sweep_results" "agentica-org/DeepScaleR-1.5B-Preview"

# Or directly with python
python instrument/sweep.py \
    --model_name_or_path "agentica-org/DeepScaleR-1.5B-Preview" \
    --data_name "aime" \
    --prompt_type "qwen-instruct" \
    --temperature 0.6 \
    --output_base_dir "./sweep_results" \
    --token_values 128 256 512 1024 2048 4096 8192
```

### Option 3: Interactive Mode

Use the interactive script for guided usage:

```bash
./run_instrumented.sh
```

### Option 4: Process Existing Telemetry Logs

If you have existing tegrastats logs to process:

```bash
python instrument/telemetry_proc.py \
    --log_file "tegrastats.log" \
    --output_dir "./processed_results" \
    --config_suffix "custom"
```

## 📈 Output Files

The system generates several output files for comprehensive analysis:

### Performance Metrics (`performance_*.csv`)
```csv
timestamp,model,question_id,ttft,decode_time,total_time,tokens_generated,tokens_per_second,prompt_tokens,completion_tokens,batch_total_time
2025-06-20T10:30:00,model,0,0.234,1.456,1.690,45,30.82,156,45,12.34
```

### Energy & Telemetry (`energy_*.csv`)
Wide-format CSV with 50+ metrics including:
- **Memory**: RAM usage, SWAP, IRAM with LFB details
- **CPU**: Individual core usage and frequencies  
- **GPU**: Usage percentages and frequencies (single/dual GPC)
- **Power Rails**: CPU, GPU, SOC, CV, VDDRQ, SYS5V current/average
- **Temperature**: All thermal sensors (CPU, GPU, thermal zones)
- **Specialized**: EMC, VIC, APE, MTS, NVENC, NVDEC, NVDLA frequencies

### Evaluation Results (`*.jsonl`)
Standard evaluation results with generated responses and correctness metrics.

### Sweep Summary (`sweep_summary.json`)
Summary of parameter sweep results with success/failure status for each configuration.

## 🔧 Configuration Options

### Key Arguments

- `--model_name_or_path`: Path to the model
- `--data_name`: Dataset name (aime, math, etc.)
- `--prompt_type`: Prompt template type
- `--max_tokens`: Maximum tokens to generate
- `--temperature`: Sampling temperature
- `--config_suffix`: Suffix for output files (useful for organizing runs)

### Data Sources

The system supports various math/reasoning datasets:
- AIME (American Invitational Mathematics Examination)
- MATH dataset
- Custom datasets (extend in `utils/data_loader.py`)

## 🧠 Architecture

### Performance Instrumentation (`instrument/performance.py`)

Uses VLLM's native `RequestOutput` objects to extract accurate timing:
- **TTFT**: Time to first token (estimated from VLLM prefill phase)
- **Decode Time**: Token generation time
- **Tokens/Second**: Actual throughput calculation
- **Token Counts**: Prompt and completion tokens from VLLM

### Telemetry Management (`instrument/telemetry.py`)

Context manager for tegrastats:
```python
with TelemetryHandler(output_dir, config_suffix) as telemetry:
    # Your inference code here
    log_file = telemetry.get_log_file()
```

### Comprehensive Parsing (`instrument/telemetry_proc.py`)

Parses all known tegrastats metrics:
- Handles multiple format variations
- Dynamic temperature sensor detection
- Comprehensive power rail parsing
- Energy consumption calculations (1Hz sampling assumption)

### Parameter Sweeping (`instrument/sweep.py`)

Automated parameter exploration:
- Configurable parameter ranges
- Organized output directories
- Success/failure tracking
- Summary generation

## 📊 Analysis Examples

### Energy Analysis
```python
import pandas as pd

# Load energy data
df = pd.read_csv('energy_tokens_2048.csv')

# Calculate average power consumption
avg_power = df['pom_5v_in_current_mw'].mean()
total_energy = avg_power * len(df) / 1000  # Convert to Joules

print(f"Average Power: {avg_power:.1f}mW")
print(f"Total Energy: {total_energy:.2f}J")
```

### Performance Analysis
```python
import pandas as pd

# Load performance data
df = pd.read_csv('performance_tokens_2048.csv')

# Analyze timing metrics
print(f"Average TTFT: {df['ttft'].mean():.3f}s")
print(f"Average Tokens/Sec: {df['tokens_per_second'].mean():.1f}")
print(f"Average Total Time: {df['total_time'].mean():.3f}s")
```

## 🔍 Troubleshooting

### Common Issues

1. **Permission denied for tegrastats**
   ```bash
   sudo chmod +x /usr/bin/tegrastats
   # Or run with sudo if needed
   ```

2. **Import errors**
   ```bash
   # Ensure you're in the eval directory
   cd /path/to/tegra_monitor/eval
   export PYTHONPATH=$PYTHONPATH:$(pwd)
   ```

3. **GPU memory issues**
   - Adjust `gpu_memory_utilization` in eval_instrumented.py
   - Reduce batch size or max_tokens

4. **Tegrastats not logging**
   - Check if tegrastats is available: `which tegrastats`
   - Ensure proper permissions
   - Verify Jetson platform compatibility

### Debug Mode

Enable verbose logging by adding debug prints or using Python's logging module.

## 🚀 Extension Points

### Adding New Metrics

1. **Performance Metrics**: Extend `VLLMTimingMetrics` dataclass
2. **Telemetry Metrics**: Add parsing patterns in `parse_tegrastats_line()`
3. **Parameter Sweeps**: Extend `ParameterSweep` class with new sweep types

### Custom Datasets

Add support in `utils/data_loader.py` and `utils/parser.py`.

### New Output Formats

Extend the CSV writers or add JSON/Parquet export options.

## 📝 License

This project follows the same license as the parent repository.

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Add tests for new functionality
4. Submit a pull request

## 📞 Support

For issues related to:
- **VLLM**: Check VLLM documentation
- **Jetson/Tegrastats**: NVIDIA Jetson documentation
- **This instrumentation**: Create an issue with detailed logs

---

**Happy Evaluating! 🎯**

*For the most accurate energy measurements, ensure your Jetson device is in maximum performance mode and not thermal throttling.*
