# TRM Tool Calling - Complete Implementation

Simplified TRM (Tiny Recursive Model) for tool calling with multiple model sizes.

## 🎯 Model Sizes

| Size | Params | Layers | Hidden | Reasoning | Action | Use Case |
|------|--------|--------|--------|-----------|--------|----------|
| **Tiny** | 2M | 2 | 128 | 64 | 32 | Quick testing |
| **Small** | 32M | 8 | 512 | 256 | 128 | Production (recommended) |
| **Base** | 99M | 12 | 768 | 512 | 256 | High accuracy |
| **Large** | 329M | 24 | 1024 | 768 | 512 | Research |

## 🚀 Quick Start

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

### 2. Train Model

**Small model (recommended)**:
```bash
python scripts/train_with_config.py --model_size small --num_epochs 50
```

**Tiny model (for testing)**:
```bash
python scripts/quick_train.py --num_epochs 1
```

### 3. Inference

**Interactive mode**:
```bash
python scripts/inference.py \
    --model_path checkpoints/small/best_model.pt \
    --tokenizer_path checkpoints/small/sp_tokenizer.model \
    --interactive
```

**Single prediction**:
```bash
python scripts/inference.py \
    --model_path checkpoints/small/best_model.pt \
    --tokenizer_path checkpoints/small/sp_tokenizer.model \
    --tools '[{"name": "web_chain_details", ...}]' \
    --query "I need details about Ethereum blockchain"
```

## 📁 Project Structure

```
TRM/
├── configs/
│   └── model_configs.py        # Model size configurations
├── data/
│   ├── xlam_1k_swift.json      # Training data
│   └── xlam_val_200_swift.json # Validation data
├── src/
│   ├── tokenizer.py            # SentencePiece tokenizer
│   ├── simple_tokenizer.py     # Simple char tokenizer (for testing)
│   ├── dataset.py              # Dataset loader
│   ├── collator.py             # Data collator
│   └── model.py                # TRM model
├── scripts/
│   ├── train_with_config.py    # Main training script (with LR scheduler)
│   ├── quick_train.py          # Quick test training
│   ├── inference.py            # Inference script
│   └── train_tokenizer.py      # Tokenizer training only
├── checkpoints/                # Saved models
│   ├── tiny/
│   ├── small/
│   └── base/
└── README.md
```

## 🎓 Features

### ✅ Implemented

- **Multiple Model Sizes**: Tiny (2M) → Small (32M) → Base (99M) → Large (329M)
- **SentencePiece Tokenizer**: BPE/Unigram with special tokens
- **TRM Architecture**:
  - Recursive Reasoning Module (single-z approach)
  - Action State Module
  - Deep Supervision (2-8 steps)
  - Gradient detachment between steps
- **Multi-task Learning**:
  - Tool selection
  - Correctness prediction (Q)
  - Arguments generation
- **Training Features**:
  - Learning rate scheduler (warmup + cosine decay)
  - Gradient clipping
  - Mixed precision (optional)
  - Checkpointing
- **Inference**:
  - Greedy decoding
  - Interactive mode
  - Batch prediction

### ⚠️ Optional Improvements (Not Implemented)

- Deep Recursion (T parameter)
- RMSNorm instead of LayerNorm
- SwiGLU FFN
- Focal Loss
- Beam search for generation

## 📊 Training Details

### Default Hyperparameters

```python
learning_rate: 1e-4
batch_size: 8
num_epochs: 50
weight_decay: 0.01
warmup_ratio: 0.1
min_lr_ratio: 0.1
gradient_clip_norm: 1.0
```

### Loss Function

```python
total_loss = 0
for step in range(max_supervision_steps):
    total_loss += 1.0 * tool_loss
    total_loss += 0.5 * q_loss
    if last_step:
        total_loss += 2.0 * args_loss
total_loss /= max_supervision_steps
```

### Learning Rate Schedule

- **Warmup**: Linear warmup for first 10% of steps
- **Decay**: Cosine decay to 10% of initial LR
- **Formula**: `lr = min_lr + (max_lr - min_lr) * 0.5 * (1 + cos(π * progress))`

## 🔧 Advanced Usage

### Custom Model Size

```python
from configs.model_configs import CONFIGS

# Modify existing config
custom_config = CONFIGS['small'].copy()
custom_config['hidden_dim'] = 640
custom_config['num_layers'] = 10

# Or create new config
CONFIGS['custom'] = {
    'hidden_dim': 640,
    'num_layers': 10,
    'num_heads': 10,
    'reasoning_dim': 320,
    'action_dim': 160,
    'num_recursions': 3,
    'max_supervision_steps': 6,
    'dropout': 0.1,
}
```

### Resume Training

```bash
python scripts/train_with_config.py \
    --model_size small \
    --tokenizer_path checkpoints/small/sp_tokenizer.model \
    --resume_from checkpoints/small/checkpoint_epoch_20.pt
```

### Adjust Learning Rate

```bash
python scripts/train_with_config.py \
    --model_size small \
    --learning_rate 5e-5 \
    --warmup_ratio 0.05 \
    --min_lr_ratio 0.05
```

## 📈 Expected Results

### Small Model (32M params, 50 epochs)

| Metric | Expected Value |
|--------|----------------|
| Tool Selection Accuracy | 85-95% |
| Arguments Quality | Good |
| Training Time (GPU) | ~2-3 hours |
| Training Time (CPU) | ~10-15 hours |

### Base Model (99M params, 50 epochs)

| Metric | Expected Value |
|--------|----------------|
| Tool Selection Accuracy | 90-98% |
| Arguments Quality | Very Good |
| Training Time (GPU) | ~6-8 hours |
| Training Time (CPU) | ~30-40 hours |

## 🐛 Troubleshooting

### CUDA Out of Memory

```bash
# Reduce batch size
python scripts/train_with_config.py --model_size small --batch_size 4

# Or use smaller model
python scripts/train_with_config.py --model_size tiny
```

### Training Too Slow

```bash
# Use tiny model for testing
python scripts/quick_train.py --num_epochs 1

# Or reduce data
head -n 100 data/xlam_1k_swift.json > data/xlam_100.json
python scripts/train_with_config.py --train_data data/xlam_100.json
```

### SentencePiece Hangs

```bash
# Use simple tokenizer instead
python scripts/quick_train.py
```

## 📚 References

1. **TRM Paper**: "Less is More: Recursive Reasoning with Tiny Networks" (arXiv:2510.04871)
2. **SentencePiece**: https://github.com/google/sentencepiece
3. **xLAM Dataset**: https://github.com/SalesforceAIResearch/xLAM

## 📝 License

MIT License

## 🤝 Contributing

Contributions welcome! Please open an issue or PR.
