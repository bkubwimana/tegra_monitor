# VLLM MMLU Benchmark

A comprehensive MMLU evaluation system edge systems with perf and energy instrumentation.

## 🚀 Features

- **VLLM Integration**: High-performance inference with comprehensive metrics
- **Parameter Sweeping**: Automated model and token limit sweeps
- **Three Evaluation Modes**: Base (reasoning), Budget (efficient), NoReasoning (direct)
- **Telemetry Monitoring**: Real-time performance and energy tracking
- **Answer Extraction**: Multi-pattern choice extraction with confidence scoring
- **Configuration-Driven**: YAML-based evaluation configurations

## 📁 Project Structure

```
vllm-mmlu-benchmark/
├── src/
│   ├── models/              
│   │   └── vllm_model.py    
│   ├── evaluators/          
│   │   ├── base_evaluator.py        
│   │   ├── budget_evaluator.py      
│   │   └── noreasoning_evaluator.py 
│   ├── telemetry/           
│   │   └── monitor.py       
│   ├── data_loaders/            
│   │   └── mmlu_loader.py   
│   └── utils/               
│       └── answer_extraction.py 
├── configs/                 
│   ├── base.yaml           
│   ├── budget.yaml         
│   └── noreasoning.yaml    
├── tests/                  
├── all_budget.py           
├── sweep_budget.sh         
└── README.md              
```

## 🛠️ Installation

```bash
pip install vllm transformers datasets pyyaml tqdm
```

## 📊 Usage

### Parameter Sweeping

```bash
# Run automated sweep across models and token limits
./sweep_budget.sh

# Single evaluation
python3 all_budget.py --model "deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B" --max-tokens 256
```

### Configuration

Edit `sweep_budget.sh` to customize:

```bash
MODELS=(
    "deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B"
    "deepseek-ai/DeepSeek-R1-Distill-Qwen-14B"
    "deepseek-ai/DeepSeek-R1-Distill-Llama-8B"
)

MAX_TOKENS_VALUES=(128 256 512)
```

## ⚙️ Evaluation Modes

**Base**: Full reasoning with 4096 tokens
**Budget**: Efficient evaluation with configurable token limits (128-512)
**NoReasoning**: Direct answer selection with 4096 tokens

## 📈 Output Files

### Summary Results (`all_subjects_summary.json`)
```json
{
  "model": "deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B",
  "overall_accuracy": 0.319,
  "total_questions": 3000,
  "total_correct": 959,
  "config_details": {
    "model_settings": {
      "max_tokens": 128,
      "temperature": 0.6
    }
  }
}
```

### Performance Metrics (`detailed_results_*.csv`)
Per-question results with timing, token counts, and accuracy data.

### Telemetry Data (`tegrastats_*.log`, `energy_*.csv`)
System performance and energy consumption metrics.

## 🔬 Supported Models

- `deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B`
- `deepseek-ai/DeepSeek-R1-Distill-Qwen-14B`
- `deepseek-ai/DeepSeek-R1-Distill-Llama-8B`
- `l3lab/L1-Qwen-1.5B-Max`
- Custom models via command line

## 🧪 Testing

```bash
python tests/test_answer_extraction.py
python tests/test_mmlu_loader.py
```

## � Architecture

- **VLLM Integration**: Native API with performance monitoring
- **Modular Design**: Pluggable evaluators and configurations  
- **Telemetry System**: Real-time energy and performance tracking
- **Answer Extraction**: Multi-pattern matching with confidence scoring

---

**Comprehensive MMLU benchmarking with perf and energy metrics**
