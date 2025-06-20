#!/usr/bin/env python3
import os
import sys
import subprocess
import argparse
from typing import List, Dict
import json

class ParameterSweep:
    def __init__(self, base_args: Dict, output_base_dir: str):
        self.base_args = base_args
        self.output_base_dir = output_base_dir
        self.sweep_results = []
        
    def run_max_tokens_sweep(self, token_values: List[int] = None):
        """Run parameter sweep across different max_tokens values"""
        if token_values is None:
            token_values = [128, 256, 512, 1024, 2048, 4096, 8192]
            
        print(f"Starting max_tokens sweep with values: {token_values}")
        
        for max_tokens in token_values:
            print(f"\n{'='*60}")
            print(f"Running sweep with max_tokens={max_tokens}")
            print(f"{'='*60}")
            
            # Create sweep-specific output directory
            sweep_dir = os.path.join(self.output_base_dir, f"sweep_tokens_{max_tokens}")
            os.makedirs(sweep_dir, exist_ok=True)
            
            # Update args for this sweep
            current_args = self.base_args.copy()
            current_args['max_tokens'] = max_tokens
            current_args['output_dir'] = sweep_dir
            current_args['completions_save_dir'] = os.path.join(sweep_dir, 'completions')
            
            # Run evaluation with instrumentation
            success = self._run_single_evaluation(current_args, max_tokens)
            
            if success:
                self.sweep_results.append({
                    'max_tokens': max_tokens,
                    'output_dir': sweep_dir,
                    'status': 'completed'
                })
                print(f"✓ Completed sweep for max_tokens={max_tokens}")
            else:
                self.sweep_results.append({
                    'max_tokens': max_tokens,
                    'output_dir': sweep_dir,
                    'status': 'failed'
                })
                print(f"✗ Failed sweep for max_tokens={max_tokens}")
                
        self._save_sweep_summary()
        
    def _run_single_evaluation(self, args: Dict, max_tokens: int) -> bool:
        """Run a single evaluation with the given parameters"""
        try:
            # Build command
            cmd = [
                'python', 'eval_instrumented.py',
                '--model_name_or_path', args['model_name_or_path'],
                '--data_name', args['data_name'],
                '--prompt_type', args['prompt_type'],
                '--temperature', str(args['temperature']),
                '--max_tokens', str(max_tokens),
                '--n_sampling', str(args['n_sampling']),
                '--k', str(args['k']),
                '--split', args['split'],
                '--seed', str(args['seed']),
                '--top_p', str(args['top_p']),
                '--output_dir', args['output_dir'],
                '--completions_save_dir', args['completions_save_dir'],
                '--config_suffix', f"tokens_{max_tokens}"
            ]
            
            if args.get('surround_with_messages', False):
                cmd.append('--surround_with_messages')
                
            if args.get('use_few_shot', False):
                cmd.append('--use_few_shot')
                
            # Run the command
            result = subprocess.run(cmd, capture_output=True, text=True, 
                                  cwd=os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
            
            if result.returncode == 0:
                print(f"Evaluation completed successfully")
                return True
            else:
                print(f"Evaluation failed with return code {result.returncode}")
                print(f"STDERR: {result.stderr}")
                return False
                
        except Exception as e:
            print(f"Error running evaluation: {e}")
            return False
            
    def _save_sweep_summary(self):
        """Save summary of all sweep results"""
        summary_file = os.path.join(self.output_base_dir, 'sweep_summary.json')
        
        with open(summary_file, 'w') as f:
            json.dump({
                'sweep_type': 'max_tokens',
                'base_args': self.base_args,
                'results': self.sweep_results,
                'completed_count': len([r for r in self.sweep_results if r['status'] == 'completed']),
                'failed_count': len([r for r in self.sweep_results if r['status'] == 'failed'])
            }, f, indent=2)
            
        print(f"\nSweep summary saved to: {summary_file}")
        
        # Print final summary
        completed = [r for r in self.sweep_results if r['status'] == 'completed']
        failed = [r for r in self.sweep_results if r['status'] == 'failed']
        
        print(f"\n{'='*60}")
        print(f"SWEEP SUMMARY")
        print(f"{'='*60}")
        print(f"Total runs: {len(self.sweep_results)}")
        print(f"Completed: {len(completed)}")
        print(f"Failed: {len(failed)}")
        
        if completed:
            print(f"\nCompleted configurations:")
            for result in completed:
                print(f"  - max_tokens={result['max_tokens']} -> {result['output_dir']}")

def main():
    parser = argparse.ArgumentParser(description='Run parameter sweep for model evaluation')
    parser.add_argument('--model_name_or_path', required=True, help='Model path')
    parser.add_argument('--data_name', default='aime', help='Dataset name')
    parser.add_argument('--prompt_type', default='qwen-instruct', help='Prompt type')
    parser.add_argument('--temperature', type=float, default=0.6, help='Temperature')
    parser.add_argument('--n_sampling', type=int, default=1, help='Number of samples')
    parser.add_argument('--k', type=int, default=1, help='k for pass@k')
    parser.add_argument('--split', default='test2024', help='Data split')
    parser.add_argument('--seed', type=int, default=0, help='Random seed')
    parser.add_argument('--top_p', type=float, default=0.95, help='Top-p value')
    parser.add_argument('--surround_with_messages', action='store_true', help='Use chat template')
    parser.add_argument('--use_few_shot', action='store_true', help='Use few shot examples')
    parser.add_argument('--output_base_dir', required=True, help='Base output directory for sweep')
    parser.add_argument('--token_values', nargs='+', type=int, 
                       default=[128, 256, 512, 1024, 2048, 4096, 8192],
                       help='Max token values to sweep')
    
    args = parser.parse_args()
    
    # Convert args to dict
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
        'use_few_shot': args.use_few_shot
    }
    
    # Create output directory
    os.makedirs(args.output_base_dir, exist_ok=True)
    
    # Run sweep
    sweep = ParameterSweep(base_args, args.output_base_dir)
    sweep.run_max_tokens_sweep(args.token_values)

if __name__ == "__main__":
    main()
