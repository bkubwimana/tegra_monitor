import os
from datetime import datetime

class Summary:
    def __init__(self, args, model_name):
        self.args = args
        self.model_name = model_name
        self.results = {}
        
    def add_results(self, total_questions, correct_count, pass_at_k_score=None):
        self.results = {
            'total_questions': total_questions,
            'correct_count': correct_count,
            'accuracy': correct_count / total_questions if total_questions > 0 else 0,
            'pass_at_k_score': pass_at_k_score
        }
    
    def save(self, output_file_path):
        summary_dir = os.path.dirname(output_file_path)
        summary_file = os.path.join(summary_dir, 'summary.txt')
        
        with open(summary_file, 'w') as f:
            f.write(f"Evaluation Summary - {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write("=" * 50 + "\n\n")
            
            # Config
            f.write("Configuration:\n")
            f.write(f"Model: {self.model_name}\n")
            f.write(f"Dataset: {self.args.data_name} ({self.args.split})\n")
            f.write(f"Temperature: {self.args.temperature}\n")
            f.write(f"Max Tokens: {self.args.max_tokens}\n")
            f.write(f"N Sampling: {self.args.n_sampling}\n")
            f.write(f"Prompt Type: {self.args.prompt_type}\n\n")
            
            # Results
            f.write("Results:\n")
            f.write(f"Total Questions: {self.results['total_questions']}\n")
            f.write(f"Correct: {self.results['correct_count']}\n")
            f.write(f"Accuracy: {self.results['accuracy']:.4f}\n")
            if self.results['pass_at_k_score'] is not None:
                f.write(f"Pass@{self.args.k}: {self.results['pass_at_k_score']:.4f}\n")
        
        print(f"Summary saved to: {summary_file}")