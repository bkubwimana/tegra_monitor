# AIME-Preview

🚀 Real-time evaluation platform for mathematical reasoning models, featuring immediate results on AIME 2025 Part 1 (released Feb 8, 2025).

## Latest Results

### AIME 2025 Part 1 Results
> Released within 24 hours of official exam release

[Results Table/Graph will be inserted here]

### Historical Performance
- AIME 2024
- AIME 2025 Part 1 
- AIME 2025 Part 2 (Coming Soon)

## Models Under Evaluation

### Open Models
- DeepSeek Series
  - DeepSeek-R1
  - DeepSeek-R1-Distill-Qwen (1.5B, 7B, 14B, 32B)
  - DeepSeek-R1-Distill-Llama (8B, 70B)
- O Series
  - o1-preview
  - o1-mini
  - o3-mini (low/medium/high)
- Others
  - gemini-2.0-flash-thinking
  - s1
  - limo
  - QwQ

## Evaluation Protocol

### Hyperparameters
We maintain strict consistency across all evaluations:
```
{
    "temperature": 0.3,      # Controls randomness in generation
    "n_sampling": 8,         # Number of samples per question
    "max_tokens": 32768,     # Maximum response length
    "seed": 0,              # Fixed seed for reproducibility
    "top_p": 0.95           # Nucleus sampling parameter
}
