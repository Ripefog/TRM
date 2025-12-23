# Run full training pipeline for TRM Tool Calling (Windows PowerShell)

Write-Host "==========================================" -ForegroundColor Cyan
Write-Host "TRM Tool Calling - Full Training Pipeline" -ForegroundColor Cyan
Write-Host "==========================================" -ForegroundColor Cyan

# 1. Install dependencies
Write-Host ""
Write-Host "[Step 1/4] Checking dependencies..." -ForegroundColor Yellow
pip install datasets tokenizers tqdm --quiet

# 2. Download and format data
Write-Host ""
Write-Host "[Step 2/4] Downloading and formatting Hermes dataset..." -ForegroundColor Yellow
python scripts/format_hermes_data.py `
    --train_size 10000 `
    --val_size 1000 `
    --output_dir data/processed

if ($LASTEXITCODE -ne 0) {
    Write-Host "Error: Failed to format data" -ForegroundColor Red
    exit 1
}

# 3. Train tokenizer
Write-Host ""
Write-Host "[Step 3/4] Training tokenizer..." -ForegroundColor Yellow
python scripts/train_tokenizer.py `
    --data_path data/processed/hermes_train_10000.json `
    --output_path data/processed/tokenizer.json `
    --vocab_size 32000

if ($LASTEXITCODE -ne 0) {
    Write-Host "Error: Failed to train tokenizer" -ForegroundColor Red
    exit 1
}

# 4. Train model
Write-Host ""
Write-Host "[Step 4/4] Training TRM model..." -ForegroundColor Yellow
python train_tool.py `
    --config tool_calling `
    --output_dir checkpoints/trm_tool_calling `
    --num_epochs 60 `
    --batch_size 8 `
    --gradient_accumulation_steps 4

if ($LASTEXITCODE -ne 0) {
    Write-Host "Error: Training failed" -ForegroundColor Red
    exit 1
}

Write-Host ""
Write-Host "==========================================" -ForegroundColor Green
Write-Host "Training Complete!" -ForegroundColor Green
Write-Host "Checkpoints saved to: checkpoints/trm_tool_calling" -ForegroundColor Green
Write-Host "==========================================" -ForegroundColor Green
