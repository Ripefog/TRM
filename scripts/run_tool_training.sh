#!/bin/bash
# Run full training pipeline for TRM Tool Calling

set -e  # Exit on error

echo "=========================================="
echo "TRM Tool Calling - Full Training Pipeline"
echo "=========================================="

# 1. Install dependencies
echo ""
echo "[Step 1/4] Checking dependencies..."
pip install datasets tokenizers tqdm --quiet

# 2. Download and format data
echo ""
echo "[Step 2/4] Downloading and formatting Hermes dataset..."
python scripts/format_hermes_data.py \
    --train_size 10000 \
    --val_size 1000 \
    --output_dir data/processed

# 3. Train tokenizer
echo ""
echo "[Step 3/4] Training tokenizer..."
python scripts/train_tokenizer.py \
    --data_path data/processed/hermes_train_10000.json \
    --output_path data/processed/tokenizer.json \
    --vocab_size 32000

# 4. Train model
echo ""
echo "[Step 4/4] Training TRM model..."
python train_tool.py \
    --config tool_calling \
    --output_dir checkpoints/trm_tool_calling \
    --num_epochs 60 \
    --batch_size 8 \
    --gradient_accumulation_steps 4

echo ""
echo "=========================================="
echo "Training Complete!"
echo "Checkpoints saved to: checkpoints/trm_tool_calling"
echo "=========================================="
