"""
Parameter sweep automation for DGX H100 systems
Supports batching experiments and comprehensive performance analysis
"""
import os
import sys
import time
import json
import subprocess
import argparse
from typing import List, Dict, Optional
from datetime import datetime

class DGXParameterSweep:
    """Parameter sweep optimized for DGX H100 systems"""
    
    def __init__(self, base_args: Dict, output_base_dir: str):
        self.base_args = base_args
        self.output_base_dir = output_base_dir
        self.sweep_results = []
        
        # Ensure output directory exists
        os.makedirs(output_base_dir, exist_ok=True)
        
    def run_batch_size_sweep(self, batch_sizes: List[int] = None):
        """Run parameter sweep across different batch sizes"""
        if batch_sizes is None:
            batch_sizes = [1, 5, 15, 30]
            
        print(f"Starting batch size sweep with values: {batch_sizes}")
        print(f"Output directory: {self.output_base_dir}")
        
        for batch_size in batch_sizes:
            print(f"\n{'='*70}")
            print(f"Running sweep with batch_size={batch_size}")
            print(f"{'='*70}")
            
            # Create sweep-specific output directory
            sweep_dir = os.path.join(self.output_base_dir, f"sweep_batch_{batch_size}")
            os.makedirs(sweep_dir, exist_ok=True)
            
            # Update args for this sweep
            current_args = self.base_args.copy()
            current_args['batch_size'] = batch_size
            current_args['output_dir'] = sweep_dir
            current_args['completions_save_dir'] = os.path.join(sweep_dir, 'completions')
            current_args['config_suffix'] = f"batch_{batch_size}"
            
            # Adjust max_batch_size based on batch_size
            current_args['max_batch_size'] = max(batch_size, current_args.get('max_batch_size', 64))
            
            # Run evaluation
            success, metrics = self._run_single_evaluation(current_args, batch_size)
            
            result = {
                'batch_size': batch_size,
                'status': 'completed' if success else 'failed',
                'timestamp': datetime.now().isoformat(),
                'output_dir': sweep_dir,
                'metrics': metrics if success else None
            }
            
            if success:
                print(f"✓ Batch size {batch_size} completed successfully")
                if metrics:
                    print(f"  Throughput: {metrics.get('questions_per_second', 0):.2f} questions/s")
                    print(f"  Token throughput: {metrics.get('tokens_per_second', 0):.2f} tokens/s")
            else:
                print(f"✗ Batch size {batch_size} failed")
                
            self.sweep_results.append(result)
                
        self._save_sweep_summary()
        self._print_final_summary()
        
    def run_token_sweep(self, token_values: List[int] = None, batch_size: int = 32):
        """Run parameter sweep across different max_tokens values"""
        if token_values is None:
            token_values = [8192, 16384, 32768]  
            
        print(f"Starting max_tokens sweep with values: {token_values}")
        print(f"Fixed batch size: {batch_size}")
        
        for max_tokens in token_values:
            print(f"\n{'='*70}")
            print(f"Running sweep with max_tokens={max_tokens}")
            print(f"{'='*70}")
            
            # Create sweep-specific output directory
            sweep_dir = os.path.join(self.output_base_dir, f"sweep_tokens_{max_tokens}")
            os.makedirs(sweep_dir, exist_ok=True)
            
            # Update args for this sweep
            current_args = self.base_args.copy()
            current_args['max_tokens'] = max_tokens
            current_args['batch_size'] = batch_size
            current_args['output_dir'] = sweep_dir
            current_args['completions_save_dir'] = os.path.join(sweep_dir, 'completions')
            current_args['config_suffix'] = f"tokens_{max_tokens}_batch_{batch_size}"
            
            # Run evaluation
            success, metrics = self._run_single_evaluation(current_args, f"tokens_{max_tokens}")
            
            result = {
                'max_tokens': max_tokens,
                'batch_size': batch_size,
                'status': 'completed' if success else 'failed',
                'timestamp': datetime.now().isoformat(),
                'output_dir': sweep_dir,
                'metrics': metrics if success else None
            }
            
            if success:
                print(f"✓ Max tokens {max_tokens} completed successfully")
                if metrics:
                    print(f"  Throughput: {metrics.get('questions_per_second', 0):.2f} questions/s")
                    print(f"  Token throughput: {metrics.get('tokens_per_second', 0):.2f} tokens/s")
            else:
                print(f"✗ Max tokens {max_tokens} failed")
                
            self.sweep_results.append(result)
                
        self._save_sweep_summary()
        self._print_final_summary()
    
    def run_combined_sweep(self, 
                          batch_sizes: List[int] = None, 
                          token_values: List[int] = None):
        """Run combined batch size and token length sweep"""
        if batch_sizes is None:
            batch_sizes = [5, 10, 15]
        if token_values is None:
            token_values = [8192, 16384, 32768]
            
        print(f"Starting combined sweep:")
        print(f"  Batch sizes: {batch_sizes}")
        print(f"  Token values: {token_values}")
        
        for batch_size in batch_sizes:
            for max_tokens in token_values:
                print(f"\n{'='*70}")
                print(f"Running sweep with batch_size={batch_size}, max_tokens={max_tokens}")
                print(f"{'='*70}")
                
                # Create sweep-specific output directory
                sweep_dir = os.path.join(self.output_base_dir, f"sweep_b{batch_size}_t{max_tokens}")
                os.makedirs(sweep_dir, exist_ok=True)
                
                # Update args for this sweep
                current_args = self.base_args.copy()
                current_args['batch_size'] = batch_size
                current_args['max_tokens'] = max_tokens
                current_args['output_dir'] = sweep_dir
                current_args['completions_save_dir'] = os.path.join(sweep_dir, 'completions')
                current_args['config_suffix'] = f"b{batch_size}_t{max_tokens}"
                current_args['max_batch_size'] = max(batch_size, current_args.get('max_batch_size', 64))
                
                # Run evaluation
                success, metrics = self._run_single_evaluation(
                    current_args, 
                    f"batch_{batch_size}_tokens_{max_tokens}"
                )
                
                result = {
                    'batch_size': batch_size,
                    'max_tokens': max_tokens,
                    'status': 'completed' if success else 'failed',
                    'timestamp': datetime.now().isoformat(),
                    'output_dir': sweep_dir,
                    'metrics': metrics if success else None
                }
                
                if success:
                    print(f"✓ Batch {batch_size}, Tokens {max_tokens} completed successfully")
                    if metrics:
                        print(f"  Throughput: {metrics.get('questions_per_second', 0):.2f} questions/s")
                        print(f"  Token throughput: {metrics.get('tokens_per_second', 0):.2f} tokens/s")
                        print(f"  GPU avg util: {metrics.get('avg_gpu_util', 0):.1f}%")
                else:
                    print(f"✗ Batch {batch_size}, Tokens {max_tokens} failed")
                    
                self.sweep_results.append(result)
                
        self._save_sweep_summary()
        self._print_final_summary()
        
    def _run_single_evaluation(self, args: Dict, sweep_id: str) -> tuple[bool, Optional[Dict]]:
        """Run a single evaluation with the given parameters"""
        try:
            # Build command for DGX evaluation
            cmd = [
                'python', 'eval_dgx.py',
                '--model_name_or_path', args['model_name_or_path'],
                '--data_name', args['data_name'],
                '--prompt_type', args['prompt_type'],
                '--temperature', str(args['temperature']),
                '--max_tokens', str(args['max_tokens']),
                '--batch_size', str(args['batch_size']),
                '--max_batch_size', str(args.get('max_batch_size', 64)),
                '--n_sampling', str(args['n_sampling']),
                '--k', str(args['k']),
                '--split', args['split'],
                '--seed', str(args['seed']),
                '--top_p', str(args['top_p']),
                '--output_dir', args['output_dir'],
                '--completions_save_dir', args['completions_save_dir'],
                '--config_suffix', args['config_suffix'],
                '--gpu_memory_utilization', str(args.get('gpu_memory_utilization', 0.95)),
                '--telemetry_interval', str(args.get('telemetry_interval', 0.5)),
            ]
            
            # Optional arguments
            if args.get('surround_with_messages', False):
                cmd.append('--surround_with_messages')
                
            if args.get('use_few_shot', False):
                cmd.append('--use_few_shot')
                
            if args.get('tensor_parallel_size'):
                cmd.extend(['--tensor_parallel_size', str(args['tensor_parallel_size'])])
                
            if args.get('disable_telemetry', False):
                cmd.append('--disable_telemetry')
            
            print(f"Running command: {' '.join(cmd[:8])}...")  # Print first part of command
            
            # Run the evaluation
            start_time = time.time()
            result = subprocess.run(
                cmd, 
                capture_output=True, 
                text=True,
                cwd=os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                timeout=7200  # 2 hour timeout
            )\n            end_time = time.time()
            \n            if result.returncode == 0:
                print(f\"Evaluation completed in {end_time - start_time:.1f}s\")
                
                # Try to extract metrics from output
                metrics = self._extract_metrics_from_output(result.stdout, args['output_dir'])
                return True, metrics
            else:
                print(f\"Evaluation failed with return code {result.returncode}\")
                print(f\"STDERR: {result.stderr[:500]}...\")
                return False, None
                
        except subprocess.TimeoutExpired:
            print(f\"Evaluation timed out after 2 hours\")
            return False, None
        except Exception as e:
            print(f\"Error running evaluation: {e}\")
            return False, None
    
    def _extract_metrics_from_output(self, stdout: str, output_dir: str) -> Optional[Dict]:
        \"\"\"Extract performance metrics from evaluation output\"\"\"
        metrics = {}
        
        try:
            # Parse stdout for performance information
            lines = stdout.split('\\n')
            for line in lines:
                if 'Overall throughput:' in line:
                    # Extract questions per second
                    parts = line.split()
                    for i, part in enumerate(parts):
                        if part == 'questions/s' and i > 0:
                            metrics['questions_per_second'] = float(parts[i-1])
                            break
                            
                elif 'Overall token throughput:' in line:
                    # Extract tokens per second
                    parts = line.split()
                    for i, part in enumerate(parts):
                        if part == 'tokens/s' and i > 0:
                            metrics['tokens_per_second'] = float(parts[i-1])
                            break
                            
                elif 'Accuracy:' in line:
                    # Extract accuracy
                    parts = line.split()
                    for i, part in enumerate(parts):
                        if part == 'Accuracy:' and i < len(parts) - 1:
                            metrics['accuracy'] = float(parts[i+1])
                            break
            
            # Try to load additional metrics from CSV files
            performance_files = []
            for file in os.listdir(output_dir):
                if file.startswith('performance_') and file.endswith('.csv'):
                    performance_files.append(os.path.join(output_dir, file))
            
            if performance_files:
                # Load the most recent performance file
                latest_file = max(performance_files, key=os.path.getmtime)
                try:
                    import pandas as pd
                    df = pd.read_csv(latest_file)
                    if not df.empty:
                        metrics['avg_batch_duration'] = df['batch_duration'].mean()
                        metrics['total_questions'] = df['batch_size'].sum()
                        metrics['total_tokens'] = df['total_completion_tokens'].sum()
                except ImportError:
                    pass  # pandas not available
                except Exception as e:
                    print(f\"Warning: Could not parse performance CSV: {e}\")
            
            return metrics if metrics else None
            
        except Exception as e:
            print(f\"Warning: Could not extract metrics: {e}\")
            return None
    
    def _save_sweep_summary(self):
        \"\"\"Save comprehensive sweep summary\"\"\"
        summary_file = os.path.join(self.output_base_dir, 'dgx_sweep_summary.json')
        
        summary_data = {
            'sweep_info': {
                'total_runs': len(self.sweep_results),
                'completed_runs': len([r for r in self.sweep_results if r['status'] == 'completed']),
                'failed_runs': len([r for r in self.sweep_results if r['status'] == 'failed']),
                'start_time': min([r['timestamp'] for r in self.sweep_results]) if self.sweep_results else None,
                'end_time': max([r['timestamp'] for r in self.sweep_results]) if self.sweep_results else None
            },
            'base_configuration': self.base_args,
            'results': self.sweep_results
        }
        
        with open(summary_file, 'w') as f:
            json.dump(summary_data, f, indent=2, default=str)
            
        print(f\"\\nSweep summary saved to: {summary_file}\")
        
        # Also save a simplified CSV for easy analysis
        self._save_results_csv()
    
    def _save_results_csv(self):
        \"\"\"Save results in CSV format for easy analysis\"\"\"
        csv_file = os.path.join(self.output_base_dir, 'dgx_sweep_results.csv')
        
        try:
            import csv
            with open(csv_file, 'w', newline='') as f:
                writer = csv.writer(f)
                
                # Determine columns based on available data
                if self.sweep_results:
                    sample_result = self.sweep_results[0]
                    columns = ['status', 'timestamp']
                    
                    # Add parameter columns
                    if 'batch_size' in sample_result:
                        columns.append('batch_size')
                    if 'max_tokens' in sample_result:
                        columns.append('max_tokens')
                    
                    # Add metric columns
                    metric_columns = []
                    for result in self.sweep_results:
                        if result.get('metrics'):
                            metric_columns.extend(result['metrics'].keys())
                    metric_columns = list(set(metric_columns))
                    columns.extend(metric_columns)
                    
                    writer.writerow(columns)
                    
                    # Write data
                    for result in self.sweep_results:
                        row = []
                        for col in columns:
                            if col in result:
                                row.append(result[col])
                            elif result.get('metrics') and col in result['metrics']:
                                row.append(result['metrics'][col])
                            else:
                                row.append('')
                        writer.writerow(row)
            
            print(f\"Results CSV saved to: {csv_file}\")
            
        except Exception as e:
            print(f\"Warning: Could not save CSV: {e}\")
    
    def _print_final_summary(self):
        \"\"\"Print comprehensive final summary\"\"\"
        completed = [r for r in self.sweep_results if r['status'] == 'completed']
        failed = [r for r in self.sweep_results if r['status'] == 'failed']
        
        print(f\"\\n{'='*70}\")
        print(f\"DGX PARAMETER SWEEP SUMMARY\")
        print(f\"{'='*70}\")
        print(f\"Total runs: {len(self.sweep_results)}\")
        print(f\"Completed: {len(completed)}\")
        print(f\"Failed: {len(failed)}\")
        
        if completed:
            print(f\"\\nSuccessful configurations:\")
            
            # Find best performing configurations
            best_throughput = None
            best_accuracy = None
            
            for result in completed:
                metrics = result.get('metrics', {})
                
                # Display basic info
                config_str = \"\"
                if 'batch_size' in result:
                    config_str += f\"batch={result['batch_size']} \"
                if 'max_tokens' in result:
                    config_str += f\"tokens={result['max_tokens']} \"
                
                throughput = metrics.get('questions_per_second', 0)
                accuracy = metrics.get('accuracy', 0)
                
                print(f\"  {config_str}: {throughput:.2f} q/s, {accuracy:.3f} acc\")
                
                # Track best results
                if best_throughput is None or throughput > best_throughput[1]:
                    best_throughput = (result, throughput)
                if best_accuracy is None or accuracy > best_accuracy[1]:
                    best_accuracy = (result, accuracy)
            
            if best_throughput:
                result, throughput = best_throughput
                config_str = \"\"
                if 'batch_size' in result:
                    config_str += f\"batch={result['batch_size']} \"
                if 'max_tokens' in result:
                    config_str += f\"tokens={result['max_tokens']} \"
                print(f\"\\n🚀 Best throughput: {config_str}({throughput:.2f} questions/s)\")
            
            if best_accuracy and best_accuracy != best_throughput:
                result, accuracy = best_accuracy
                config_str = \"\"
                if 'batch_size' in result:
                    config_str += f\"batch={result['batch_size']} \"
                if 'max_tokens' in result:
                    config_str += f\"tokens={result['max_tokens']} \"
                print(f\"🎯 Best accuracy: {config_str}({accuracy:.4f})\")

def main():
    parser = argparse.ArgumentParser(description='DGX H100 parameter sweep for LLM evaluation')
    
    # Required arguments
    parser.add_argument('--model_name_or_path', required=True, help='Model path or name')
    parser.add_argument('--output_base_dir', required=True, help='Base output directory for sweep')
    
    # Sweep configuration
    parser.add_argument('--sweep_type', choices=['batch', 'tokens', 'combined'], 
                       default='batch', help='Type of parameter sweep')
    parser.add_argument('--batch_sizes', nargs='+', type=int, 
                       default=[1, 5, 10, 15], help='Batch sizes to sweep')
    parser.add_argument('--token_values', nargs='+', type=int, 
                       default=[8192, 16384, 32768], help='Max token values to sweep')
    
    # Base evaluation arguments
    parser.add_argument('--data_name', default='aime', help='Dataset name')
    parser.add_argument('--prompt_type', default='qwen-instruct', help='Prompt type')
    parser.add_argument('--temperature', type=float, default=0.0, help='Temperature')
    parser.add_argument('--n_sampling', type=int, default=1, help='Number of samples')
    parser.add_argument('--k', type=int, default=1, help='k for pass@k')
    parser.add_argument('--split', default='test2024', help='Data split')
    parser.add_argument('--seed', type=int, default=42, help='Random seed')
    parser.add_argument('--top_p', type=float, default=1.0, help='Top-p value')
    
    # Model configuration
    parser.add_argument('--gpu_memory_utilization', type=float, default=0.95, 
                       help='GPU memory utilization')
    parser.add_argument('--tensor_parallel_size', type=int, help='Tensor parallel size')
    parser.add_argument('--disable_telemetry', action='store_true', 
                       help='Disable GPU telemetry')
    
    # Optional flags
    parser.add_argument('--surround_with_messages', action='store_true', 
                       help='Use chat template')
    parser.add_argument('--use_few_shot', action='store_true', 
                       help='Use few shot examples')
    
    args = parser.parse_args()
    
    # Convert args to dict for base configuration
    base_args = {
        'model_name_or_path': args.model_name_or_path,
        'data_name': args.data_name,
        'prompt_type': args.prompt_type,
        'temperature': args.temperature,
        'n_sampling': args.n_sampling,
        'k': args.k,
        'split': args.split,
        'seed': args.seed,
        'top_p': args.top_p,
        'surround_with_messages': args.surround_with_messages,
        'use_few_shot': args.use_few_shot,
        'gpu_memory_utilization': args.gpu_memory_utilization,
        'disable_telemetry': args.disable_telemetry,
        'telemetry_interval': 0.5,
    }
    
    if args.tensor_parallel_size:
        base_args['tensor_parallel_size'] = args.tensor_parallel_size
    
    # Create output directory
    os.makedirs(args.output_base_dir, exist_ok=True)
    
    # Initialize and run sweep
    sweep = DGXParameterSweep(base_args, args.output_base_dir)
    
    if args.sweep_type == 'batch':
        sweep.run_batch_size_sweep(args.batch_sizes)
    elif args.sweep_type == 'tokens':
        sweep.run_token_sweep(args.token_values, batch_size=args.batch_sizes[0])
    elif args.sweep_type == 'combined':
        sweep.run_combined_sweep(args.batch_sizes, args.token_values)

if __name__ == \"__main__\":
    main()
