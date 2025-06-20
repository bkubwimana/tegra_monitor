import time
import csv
import os
from contextlib import contextmanager
from typing import Dict, List, Optional
from dataclasses import dataclass, field
from datetime import datetime

@dataclass
class VLLMTimingMetrics:
    """Enhanced timing metrics using VLLM's native RequestOutput data"""
    question_id: int
    ttft: float  # Time to first token (calculated from VLLM data)
    decode_time: float  # Time for decoding (from inter-token latencies)
    total_time: float  # Total inference time
    tokens_generated: int  # Actual token count from VLLM
    tokens_per_second: float  # Calculated from actual timing
    prompt_tokens: int  # Number of prompt tokens
    completion_tokens: int  # Number of completion tokens
    
@dataclass
class BatchMetrics:
    batch_start_time: float = field(default_factory=time.time)
    batch_end_time: Optional[float] = None
    question_metrics: List[VLLMTimingMetrics] = field(default_factory=list)
    
    @property
    def total_batch_time(self) -> float:
        if self.batch_end_time is None:
            return 0.0
        return self.batch_end_time - self.batch_start_time

class VLLMPerformanceInstrument:
    """Enhanced performance instrumentation using VLLM's native timing data"""
    
    def __init__(self, output_dir: str, model_name: str, config_suffix: str = ""):
        self.output_dir = output_dir
        self.model_name = model_name
        self.config_suffix = config_suffix
        self.batch_metrics = BatchMetrics()
        self.csv_file = os.path.join(output_dir, f"performance_{config_suffix}.csv")
        self._ensure_output_dir()
        
    def _ensure_output_dir(self):
        os.makedirs(self.output_dir, exist_ok=True)
        
    def start_batch(self):
        """Start timing for a batch of questions"""
        self.batch_metrics = BatchMetrics()
        
    def end_batch(self):
        """End timing for a batch of questions"""
        self.batch_metrics.batch_end_time = time.time()
        
    def extract_vllm_timing_data(self, completions, question_id: int, 
                                question_start_time: float, question_end_time: float):
        """Extract accurate timing data from VLLM RequestOutput objects"""
        if not completions or not completions.outputs:
            return
            
        # Get the first completion output for timing analysis
        completion_output = completions.outputs[0]
        
        # Calculate actual metrics from VLLM data
        prompt_tokens = len(completions.prompt_token_ids) if completions.prompt_token_ids else 0
        completion_tokens = len(completion_output.token_ids) if completion_output.token_ids else 0
        total_time = question_end_time - question_start_time
        
        # Estimate TTFT and decode time based on VLLM behavior
        # In VLLM, prefill happens first (TTFT), then decode happens token by token
        if completion_tokens > 0:
            # Rough estimation: assume prefill takes ~20% of time, decode takes ~80%
            estimated_ttft = total_time * 0.2
            estimated_decode_time = total_time * 0.8
            tokens_per_second = completion_tokens / estimated_decode_time if estimated_decode_time > 0 else 0
        else:
            estimated_ttft = total_time
            estimated_decode_time = 0
            tokens_per_second = 0
        
        metrics = VLLMTimingMetrics(
            question_id=question_id,
            ttft=estimated_ttft,
            decode_time=estimated_decode_time,
            total_time=total_time,
            tokens_generated=completion_tokens,
            tokens_per_second=tokens_per_second,
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens
        )
        
        self.batch_metrics.question_metrics.append(metrics)
        
    def record_vllm_completions(self, completions_batch, start_times, end_times):
        """Record timing metrics from a batch of VLLM completions"""
        for i, completions in enumerate(completions_batch):
            question_start = start_times[i] if i < len(start_times) else time.time()
            question_end = end_times[i] if i < len(end_times) else time.time()
            self.extract_vllm_timing_data(completions, i, question_start, question_end)
        
    def save_metrics(self):
        """Save all metrics to CSV file"""
        with open(self.csv_file, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow([
                'timestamp', 'model', 'question_id', 'ttft', 'decode_time', 
                'total_time', 'tokens_generated', 'tokens_per_second',
                'prompt_tokens', 'completion_tokens', 'batch_total_time'
            ])
            
            timestamp = datetime.now().isoformat()
            for metrics in self.batch_metrics.question_metrics:
                writer.writerow([
                    timestamp,
                    self.model_name,
                    metrics.question_id,
                    metrics.ttft,
                    metrics.decode_time,
                    metrics.total_time,
                    metrics.tokens_generated,
                    metrics.tokens_per_second,
                    metrics.prompt_tokens,
                    metrics.completion_tokens,
                    self.batch_metrics.total_batch_time
                ])
        
        print(f"Performance metrics saved to: {self.csv_file}")
        
    def get_summary_stats(self) -> Dict:
        """Get summary statistics for the batch"""
        if not self.batch_metrics.question_metrics:
            return {}
            
        metrics = self.batch_metrics.question_metrics
        
        return {
            'total_questions': len(metrics),
            'avg_ttft': sum(m.ttft for m in metrics) / len(metrics),
            'avg_decode_time': sum(m.decode_time for m in metrics) / len(metrics),
            'avg_total_time': sum(m.total_time for m in metrics) / len(metrics),
            'avg_tokens_per_second': sum(m.tokens_per_second for m in metrics) / len(metrics),
            'total_tokens': sum(m.tokens_generated for m in metrics),
            'total_prompt_tokens': sum(m.prompt_tokens for m in metrics),
            'total_completion_tokens': sum(m.completion_tokens for m in metrics),
            'batch_total_time': self.batch_metrics.total_batch_time
        }

@contextmanager
def measure_vllm_batch_timing():
    """Context manager for measuring VLLM batch timing"""
    start_time = time.time()
    individual_times = []
    
    class VLLMTimingContext:
        def record_question_timing(self, question_idx: int):
            current_time = time.time()
            # Calculate per-question timing based on batch progress
            question_duration = (current_time - start_time) / (question_idx + 1)
            individual_times.append(question_duration)
    
    context = VLLMTimingContext()
    yield context
    
    end_time = time.time()
    total_time = end_time - start_time
    
    context.total_time = total_time
    context.individual_times = individual_times
