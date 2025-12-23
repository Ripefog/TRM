# TinyRecursiveModels

Implementation of "Recursive Reasoning with Tiny Networks" from scratch.

## Project Structure
- `src/`: Source code for the model, dataset, and training logic.
- `scripts/`: Shell scripts for training and tokenization.
- `data/`: Data directory.
    - `processed/sample_1k.csv`: A small sample dataset (1k rows) for testing/debugging.

## Setup

1. Install dependencies:
```bash
pip install -r requirements.txt
```

## Usage (Colab / Local)

1. **Train Tokenizer** (Required for the first time):
   This will train a BPE tokenizer on the sample dataset.
```bash
./scripts/train_tokenizer.sh
```

2. **Train Model**:
   This will start training the Tiny Recursive Model.
```bash
./scripts/run_train.sh
```

## Configuration
You can modify model and training parameters in `src/config.py`.
Default configuration is set to use `data/processed/sample_1k.csv`.
