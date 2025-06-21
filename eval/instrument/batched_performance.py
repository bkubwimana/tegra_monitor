"""
Enhanced performance instrumentation for DGX H100 systems with batched inference
Leverages VLLM's native timing capabilities and supports larger batch sizes
"""
import os
import time
import csv
from contextlib import contextmanager
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass, field
from datetime import datetime
import statistics

@dataclass
class BatchedVLLMTimingMetrics:
    """Enhanced timing metrics for batched VLLM inference on H100"""
    batch_id: int
    batch_size: int
    batch_start_time: float
    batch_end_time: float
    batch_duration: float
    total_prompt_tokens: int
    total_completion_tokens: int
    avg_tokens_per_second: float
    median_tokens_per_second: float
    questions: List[Dict] = field(default_factory=list)  # Per-question metrics

@dataclass 
class QuestionMetrics:
    """Per-question timing metrics within a batch"""
    question_id: int
    prompt_tokens: int
    completion_tokens: int
    estimated_ttft: float  # Time to first token (estimated)
    estimated_decode_time: float  # Decoding time (estimated)
    tokens_per_second: float

class VLLMBatchedPerformanceInstrument:
    """
    Performance instrumentation optimized for high-throughput batched inference on DGX systems
    """
    
    def __init__(self, output_dir: str, model_name: str, config_suffix: str = ""):
        self.output_dir = output_dir
        self.model_name = model_name
        self.config_suffix = config_suffix
        self.batch_metrics: List[BatchedVLLMTimingMetrics] = []
        self.csv_file = os.path.join(output_dir, f"performance_batch_{config_suffix}.csv")
        self.detailed_csv_file = os.path.join(output_dir, f"performance_detailed_{config_suffix}.csv")
        self._ensure_output_dir()
        
    def _ensure_output_dir(self):
        os.makedirs(self.output_dir, exist_ok=True)
        
    def record_batch_inference(self, 
                             batch_id: int,
                             completions: List,
                             batch_start_time: float,
                             batch_end_time: float,
                             question_ids: Optional[List[int]] = None) -> BatchedVLLMTimingMetrics:
        """
        Record comprehensive timing metrics for a batch of VLLM completions
        
        Args:
            batch_id: Unique identifier for this batch
            completions: List of VLLM RequestOutput objects
            batch_start_time: Start timestamp for the batch
            batch_end_time: End timestamp for the batch
            question_ids: Optional list of question IDs corresponding to completions
        """
        batch_duration = batch_end_time - batch_start_time
        batch_size = len(completions)
        
        if question_ids is None:
            question_ids = list(range(batch_size))
        
        # Aggregate token counts
        total_prompt_tokens = 0
        total_completion_tokens = 0
        question_metrics = []
        tokens_per_second_list = []
        
        for i, completion in enumerate(completions):
            if not completion or not completion.outputs:
                continue

            # Extract token counts from VLLM RequestOutput
            try:
                prompt_tokens = len(completion.prompt_token_ids) if hasattr(completion, 'prompt_token_ids') and completion.prompt_token_ids else 0
                completion_tokens = len(completion.outputs[0].token_ids) if completion.outputs[0].token_ids else len(completion.outputs[0].text.split())
                
                total_prompt_tokens += prompt_tokens
                total_completion_tokens += completion_tokens
                
                # Try to extract precise timing metrics from VLLM
                ttft = 0
                decode_time = 0
                total_time = batch_duration / batch_size  # Fallback to estimated time per question
                
                if hasattr(completion, 'metrics') and completion.metrics:
                    m = completion.metrics
                    try:
                        ttft = m.first_token_time - m.arrival_time if hasattr(m, 'first_token_time') and hasattr(m, 'arrival_time') else 0
                        decode_time = m.last_token_time - m.first_token_time if hasattr(m, 'last_token_time') and hasattr(m, 'first_token_time') else 0
                        total_time = m.finished_time - m.arrival_time if hasattr(m, 'finished_time') and hasattr(m, 'arrival_time') else total_time
                    except AttributeError:
                        pass  # Use fallback values
                
                tokens_per_second = completion_tokens / total_time if total_time > 0 else 0
                
                tokens_per_second_list.append(tokens_per_second)
                question_metrics.append({
                    'question_id': question_ids[i] if i < len(question_ids) else i,
                    'prompt_tokens': prompt_tokens,
                    'completion_tokens': completion_tokens,
                    'ttft': ttft,
                    'decode_time': decode_time,
                    'total_time': total_time,
                    'tokens_per_second': tokens_per_second
                })
                
            except Exception as e:
                print(f"Warning: Could not extract metrics for completion {i}: {e}")
                # Add minimal entry for this question
                question_metrics.append({
                    'question_id': question_ids[i] if i < len(question_ids) else i,
                    'prompt_tokens': 0,
                    'completion_tokens': 0,
                    'ttft': 0,
                    'decode_time': 0,
                    'total_time': 0,
                    'tokens_per_second': 0
                })
        
        # Calculate aggregate metrics
        avg_tokens_per_second = total_completion_tokens / batch_duration if batch_duration > 0 else 0
        median_tokens_per_second = statistics.median(tokens_per_second_list) if tokens_per_second_list else 0
        
        batch_metrics = BatchedVLLMTimingMetrics(
            batch_id=batch_id,
            batch_size=batch_size,
            batch_start_time=batch_start_time,
            batch_end_time=batch_end_time,
            batch_duration=batch_duration,
            total_prompt_tokens=total_prompt_tokens,
            total_completion_tokens=total_completion_tokens,
            avg_tokens_per_second=avg_tokens_per_second,
            median_tokens_per_second=median_tokens_per_second,
            questions=question_metrics
        )
        
        self.batch_metrics.append(batch_metrics)
        return batch_metrics
    
    def save_metrics(self):
        """Save batch and detailed metrics to CSV files"""
        if not self.batch_metrics:
            print("No performance metrics to save")
            return
        
        # Save batch-level metrics
        self._save_batch_metrics()
        
        # Save detailed per-question metrics
        self._save_detailed_metrics()
        
        print(f"Performance metrics saved:")
        print(f"  Batch metrics: {self.csv_file}")
        print(f"  Detailed metrics: {self.detailed_csv_file}")
    
    def _save_batch_metrics(self):
        """Save batch-level performance metrics"""
        with open(self.csv_file, 'w', newline='') as f:
            writer = csv.writer(f)
            
            # Header
            header = [
                'timestamp', 'datetime', 'model', 'batch_id', 'batch_size',
                'batch_duration', 'total_prompt_tokens', 'total_completion_tokens',
                'avg_tokens_per_second', 'median_tokens_per_second',
                'throughput_questions_per_second', 'avg_prompt_tokens_per_question',
                'avg_completion_tokens_per_question'
            ]
            writer.writerow(header)
            
            # Data
            for batch in self.batch_metrics:
                dt_str = datetime.fromtimestamp(batch.batch_start_time).strftime('%Y-%m-%d %H:%M:%S.%f')[:-3]
                throughput_qps = batch.batch_size / batch.batch_duration if batch.batch_duration > 0 else 0
                avg_prompt_tokens = batch.total_prompt_tokens / batch.batch_size if batch.batch_size > 0 else 0
                avg_completion_tokens = batch.total_completion_tokens / batch.batch_size if batch.batch_size > 0 else 0
                
                row = [
                    batch.batch_start_time, dt_str, self.model_name,
                    batch.batch_id, batch.batch_size, batch.batch_duration,
                    batch.total_prompt_tokens, batch.total_completion_tokens,
                    batch.avg_tokens_per_second, batch.median_tokens_per_second,
                    throughput_qps, avg_prompt_tokens, avg_completion_tokens
                ]
                writer.writerow(row)
    
    def _save_detailed_metrics(self):
        """Save detailed per-question metrics"""
        with open(self.detailed_csv_file, 'w', newline='') as f:
            writer = csv.writer(f)
            
            # Header:
            header = [
                'batch_id', 'question_id', 'prompt_tokens', 'completion_tokens',
                'ttft', 'decode_time', 'total_time', 'tokens_per_second'
            ]
            writer.writerow(header)
            
            # Data
            for batch in self.batch_metrics:
                for question in batch.questions:
                    row = [
                        batch.batch_id,
                        question['question_id'],
                        question['prompt_tokens'],
                        question['completion_tokens'],
                        question['ttft'],
                        question['decode_time'],
                        question['total_time'],
                        question['tokens_per_second']
                    ]
                    writer.writerow(row)
    
    def get_summary_stats(self) -> Dict:
        """Get comprehensive summary statistics"""
        if not self.batch_metrics:
            return {
                'total_batches': 0,
                'total_questions': 0,
                'total_tokens': 0,
            }
        
        # Aggregate statistics across all batches
        total_batches = len(self.batch_metrics)
        total_questions = sum(batch.batch_size for batch in self.batch_metrics)
        total_prompt_tokens = sum(batch.total_prompt_tokens for batch in self.batch_metrics)
        total_completion_tokens = sum(batch.total_completion_tokens for batch in self.batch_metrics)
        total_duration = sum(batch.batch_duration for batch in self.batch_metrics)
        
        # Calculate aggregate throughput metrics
        overall_tokens_per_second = total_completion_tokens / total_duration if total_duration > 0 else 0
        overall_questions_per_second = total_questions / total_duration if total_duration > 0 else 0
        
        # Get distribution statistics
        batch_durations = [batch.batch_duration for batch in self.batch_metrics]
        tokens_per_second_list = [batch.avg_tokens_per_second for batch in self.batch_metrics]
        
        return {
            'total_batches': total_batches,
            'total_questions': total_questions,
            'total_prompt_tokens': total_prompt_tokens,
            'total_completion_tokens': total_completion_tokens,
            'total_tokens': total_prompt_tokens + total_completion_tokens,
            'total_duration': total_duration,
            'overall_tokens_per_second': overall_tokens_per_second,
            'overall_questions_per_second': overall_questions_per_second,
            'avg_batch_duration': statistics.mean(batch_durations) if batch_durations else 0,
            'median_batch_duration': statistics.median(batch_durations) if batch_durations else 0,
            'avg_tokens_per_second': statistics.mean(tokens_per_second_list) if tokens_per_second_list else 0,
            'median_tokens_per_second': statistics.median(tokens_per_second_list) if tokens_per_second_list else 0,
            'avg_prompt_tokens_per_question': total_prompt_tokens / total_questions if total_questions > 0 else 0,
            'avg_completion_tokens_per_question': total_completion_tokens / total_questions if total_questions > 0 else 0
        }

@contextmanager
def measure_batch_inference():
    """
    Context manager for measuring batched VLLM inference timing
    
    Usage:
        with measure_batch_inference() as timer:
            completions = llm.generate(prompts, sampling_params)
            timer.record_completion(completions)
    """
    start_time = time.time()
    
    class BatchTimingContext:
        def __init__(self):
            self.start_time = start_time
            self.end_time = None
            self.completions = None
        
        def record_completion(self, completions):
            self.end_time = time.time()
            self.completions = completions
            self.duration = self.end_time - self.start_time
    
    context = BatchTimingContext()
    yield context
    
    if context.end_time is None:
        context.end_time = time.time()
        context.duration = context.end_time - context.start_time
