"""
Model configuration helper for DGX systems
Handles model inspection and optimal tensor parallel size selection
"""
import os
from typing import Optional, Tuple
from transformers import AutoConfig
import math

class ModelConfigHelper:
    """Helper class for model configuration and tensor parallel optimization"""
    
    def __init__(self, model_name_or_path: str):
        self.model_name_or_path = model_name_or_path
        self._config = None
        self._attention_heads = None
        
    @property
    def config(self):
        """Lazy load model configuration"""
        if self._config is None:
            try:
                self._config = AutoConfig.from_pretrained(self.model_name_or_path, trust_remote_code=True)
            except Exception as e:
                print(f"Warning: Could not load model config: {e}")
                self._config = None
        return self._config
    
    @property
    def attention_heads(self) -> Optional[int]:
        """Get number of attention heads from model config"""
        if self._attention_heads is not None:
            return self._attention_heads
            
        if self.config is None:
            return None
            
        # Try different attribute names for attention heads
        head_attrs = ['n_head', 'num_attention_heads', 'num_heads', 'n_heads']
        
        for attr in head_attrs:
            if hasattr(self.config, attr):
                self._attention_heads = getattr(self.config, attr)
                return self._attention_heads
        
        print("Warning: Could not determine number of attention heads")
        return None
    
    def get_optimal_tensor_parallel_size(self, available_gpus: int, max_tp_size: Optional[int] = None) -> int:
        """
        Determine optimal tensor parallel size based on model architecture and available GPUs
        
        Args:
            available_gpus: Number of available GPUs
            max_tp_size: Maximum tensor parallel size to consider
            
        Returns:
            Optimal tensor parallel size
        """
        if max_tp_size is None:
            max_tp_size = available_gpus
        
        # Get number of attention heads
        num_heads = self.attention_heads
        
        if num_heads is None:
            print(f"Warning: Could not determine attention heads, using tensor_parallel_size=1")
            return 1
        
        print(f"Model has {num_heads} attention heads")
        
        # Find valid tensor parallel sizes (divisors of num_heads)
        valid_sizes = []
        for tp_size in range(1, min(max_tp_size, available_gpus) + 1):
            if num_heads % tp_size == 0:
                valid_sizes.append(tp_size)
        
        if not valid_sizes:
            print(f"Warning: No valid tensor parallel sizes found, using 1")
            return 1
        
        # Choose the largest valid size for best performance
        optimal_size = max(valid_sizes)
        
        print(f"Valid tensor parallel sizes: {valid_sizes}")
        print(f"Selected tensor parallel size: {optimal_size}")
        
        return optimal_size
    
    def get_model_info(self) -> dict:
        """Get comprehensive model information"""
        info = {
            'model_name': self.model_name_or_path,
            'attention_heads': self.attention_heads,
            'config_available': self.config is not None
        }
        
        if self.config:
            # Add common config attributes
            attrs_to_check = [
                'vocab_size', 'hidden_size', 'intermediate_size', 
                'num_hidden_layers', 'max_position_embeddings',
                'model_type', 'architectures'
            ]
            
            for attr in attrs_to_check:
                if hasattr(self.config, attr):
                    info[attr] = getattr(self.config, attr)
        
        return info
    
    @staticmethod
    def validate_tensor_parallel_config(num_heads: int, tensor_parallel_size: int) -> bool:
        """Validate if tensor parallel configuration is valid"""
        if num_heads % tensor_parallel_size != 0:
            return False
        return True
    
    @staticmethod
    def suggest_alternative_configs(num_heads: int, max_gpus: int) -> list:
        """Suggest alternative tensor parallel configurations"""
        suggestions = []
        
        for tp_size in range(1, max_gpus + 1):
            if num_heads % tp_size == 0:
                suggestions.append({
                    'tensor_parallel_size': tp_size,
                    'heads_per_gpu': num_heads // tp_size,
                    'gpus_used': tp_size
                })
        
        return suggestions
