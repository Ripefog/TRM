# TRM Tool Calling - Full Training Pipeline

## 📋 Overview

This guide walks you through training a **Tiny Recursion Model (TRM)** for **Tool Calling** using the Hermes Reasoning Tool Use dataset.

### Architecture: TRM Single-Z

```
Input x → Embedding
        ↓
┌──────────────────────────────────────────┐
│     Deep Supervision (N_sup = 16)        │
│                                          │
│   ┌──────────────────────────────────┐   │
│   │   Deep Recursion (T = 3)         │   │
│   │                                  │   │
│   │   T-1 times (no grad):           │   │
│   │     z = latent_recursion(x, z)   │   │
│   │                                  │   │
│   │   1 time (with grad):            │   │
│   │     z = latent_recursion(x, z)   │   │
│   │                                  │   │
│   │   ┌────────────────────────┐     │   │
│   │   │ Latent Recursion (n=6) │     │   │
│   │   │   for i in range(n+1): │     │   │
│   │   │     z = net(x + z)     │     │   │
│   │   └────────────────────────┘     │   │
│   └──────────────────────────────────┘   │
│                                          │
│   z = z.detach()                         │
│   if q_hat > 0.5: break (ACT)            │
└──────────────────────────────────────────┘
        ↓
Output Head(z) → logits → tool_call
Q Head(z) → halting probability
```

**Effective depth per supervision step:** T × (n+1) × n_layers = 3 × 7 × 2 = **42 layers**

## 🚀 Quick Start

### 1. Install Dependencies

```bash
pip install torch datasets tokenizers tqdm
```

### 2. Download & Format Data

```bash
# Download from HuggingFace and convert to SWIFT format
python scripts/format_hermes_data.py --train_size 10000 --val_size 1000

# Preview samples
python scripts/format_hermes_data.py --preview
```

### 3. Train Tokenizer

```bash
python scripts/train_tokenizer.py \
    --data_path data/processed/hermes_train_10000.json \
    --output_path data/processed/tokenizer.json \
    --vocab_size 32000
```

### 4. Train Model

```bash
# Default config (recommended)
python train_tool.py --config tool_calling

# Custom config
python train_tool.py \
    --d_model 256 \
    --n_heads 4 \
    --n_layers 2 \
    --n_latent_steps 6 \
    --n_recursion_steps 3 \
    --n_supervision_steps 16 \
    --batch_size 8 \
    --learning_rate 1e-4 \
    --num_epochs 60

# Resume training
python train_tool.py --resume checkpoints/checkpoint_latest.pt
```

## 📁 Project Structure

```
TinyRecursiveModels/
├── src/
│   ├── config_tool.py      # Configuration classes
│   ├── dataset_tool.py     # HermesToolDataset
│   ├── model_single_z.py   # TRM Single-Z model
│   └── trainer_tool.py     # Training loop with deep supervision
├── scripts/
│   ├── format_hermes_data.py   # Data preprocessing
│   └── train_tokenizer.py      # Tokenizer training
├── data/
│   └── processed/
│       ├── hermes_train_10000.json
│       ├── hermes_val_1000.json
│       └── tokenizer.json
├── train_tool.py           # Main training script
└── TRAINING_GUIDE.md       # This file
```

## 📊 Data Format

### Input: Hermes/ShareGPT Format
```json
{
  "conversations": [
    {"from": "system", "value": "...<tools>...</tools>..."},
    {"from": "user", "value": "What's the weather?"},
    {"from": "assistant", "value": "<think>...</think>\n<tool_call>...</tool_call>"},
    {"from": "tool", "value": "{\"temp\": 22}"},
    {"from": "assistant", "value": "The temperature is 22°C"}
  ]
}
```

### Output: SWIFT Agent Format
```json
{
  "tools": "[{\"type\": \"function\", ...}]",
  "messages": [
    {"role": "user", "content": "What's the weather?"},
    {"role": "assistant", "content": "<think>...</think>"},
    {"role": "tool_call", "content": "{\"name\": \"get_weather\", \"arguments\": {...}}"},
    {"role": "tool_response", "content": "{\"temp\": 22}"},
    {"role": "assistant", "content": "The temperature is 22°C"}
  ]
}
```

### Loss Masking Strategy

| Role | Mask | Loss Weight |
|------|------|-------------|
| System (tools) | ✓ MASKED | 0.0 |
| User | ✓ MASKED | 0.0 |
| Tool Response | ✓ MASKED | 0.0 |
| Assistant (think) | TRAINED | 1.0 |
| Assistant (tool_call) | TRAINED | **2.0** |
| Assistant (response) | TRAINED | 1.0 |

## ⚙️ Configuration

### Model Config (from TRM paper)

| Parameter | Value | Description |
|-----------|-------|-------------|
| d_model | 256 | Hidden dimension |
| n_heads | 4 | Attention heads |
| n_layers | **2** | Layers (tiny!) |
| n_latent_steps (n) | 6 | Z update iterations |
| n_recursion_steps (T) | 3 | Deep recursion loops |
| n_supervision_steps | 16 | Max supervision steps |

### Training Config

| Parameter | Value | Description |
|-----------|-------|-------------|
| batch_size | 8 | Per-GPU batch size |
| learning_rate | 1e-4 | Main LR |
| embedding_lr | 1e-2 | 100x higher for embeddings |
| weight_decay | 1.0 | Heavy regularization |
| warmup_steps | 2000 | LR warmup |
| use_ema | True | EMA for stability |
| use_act | True | Adaptive Computation Time |

## 🔑 Key Components

### 1. TRM Single-Z Model (`model_single_z.py`)

```python
class TRMSingleZ:
    def latent_recursion(self, x, z, mask):
        """z = net(x + z), repeated n+1 times"""
        for _ in range(self.n_latent_steps + 1):
            z = self.net(x + z, mask)
        return z
    
    def deep_recursion(self, x, z, mask):
        """T loops: T-1 without grad, 1 with grad"""
        with torch.no_grad():
            for _ in range(T - 1):
                z = self.latent_recursion(x, z, mask)
        z = self.latent_recursion(x, z, mask)
        return z.detach(), self.output_head(z), self.q_head(z)
```

### 2. Deep Supervision Training (`trainer_tool.py`)

```python
for step in range(N_supervision_steps):
    z, logits, q_hat = model(input_ids, z_init=z)
    
    # Main loss
    loss = cross_entropy(logits, labels)
    
    # ACT loss (halting)
    target = (predictions == labels).float()
    loss += bce(q_hat, target)
    
    loss.backward()
    optimizer.step()
    
    z = z.detach()  # Detach for next step
    
    if q_hat.mean() > 0.5:  # Early stopping
        break
```

### 3. EMA (Exponential Moving Average)

```python
class EMA:
    def update(self):
        shadow = decay * shadow + (1 - decay) * weights
```

## 📈 Expected Results

Based on TRM paper:

| Dataset | HRM (27M) | TRM (7M) |
|---------|-----------|----------|
| Sudoku-Extreme | 74% | **87.4%** |
| Maze-Hard | 74.5% | **85.3%** |
| ARC-AGI-1 | 40.3% | **44.6%** |
| ARC-AGI-2 | 5.0% | **7.8%** |

## 🔧 Troubleshooting

### Out of Memory
- Reduce `batch_size`
- Reduce `max_seq_len`
- Increase `gradient_accumulation_steps`

### Training Instability
- Ensure `use_ema=True`
- Reduce `learning_rate`
- Increase `warmup_steps`

### Poor Convergence
- Increase `n_supervision_steps`
- Increase `num_epochs`
- Check data quality

## 📚 References

- [TRM Paper](https://arxiv.org/abs/2510.04871)
- [Hermes Dataset](https://huggingface.co/datasets/interstellarninja/hermes_reasoning_tool_use)
- [SWIFT Agent Format](https://swift.readthedocs.io/en/latest/Instruction/Agent-support.html)
