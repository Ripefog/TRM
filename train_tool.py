"""
Main training script for TRM Tool Calling.

Usage:
    # Download and format data first
    python scripts/format_hermes_data.py --train_size 10000 --val_size 1000
    
    # Train tokenizer
    python scripts/train_tokenizer.py --data_path data/processed/hermes_train_10000.json
    
    # Train model
    python train_tool.py --config tool_calling
    
    # Resume training
    python train_tool.py --resume checkpoints/checkpoint_latest.pt
"""

import argparse
import os
import sys
import json
import torch
import random
import numpy as np

# Add src to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from src.config_tool import Config, TRMVariant, TOOL_CALLING_CONFIG
from src.model_single_z import TRMSingleZ, create_trm_single_z
from src.dataset_tool import HermesToolDataset, create_dataloaders
from src.trainer_tool import TRMToolTrainer

def set_seed(seed: int):
    """Set random seed for reproducibility"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

def get_tokenizer_vocab_size(tokenizer_path: str) -> int:
    """Get vocab size from tokenizer"""
    from tokenizers import Tokenizer
    tokenizer = Tokenizer.from_file(tokenizer_path)
    return tokenizer.get_vocab_size()

def main():
    parser = argparse.ArgumentParser(description="Train TRM for Tool Calling")
    
    # Config
    parser.add_argument("--config", type=str, default="tool_calling",
                       choices=["tool_calling", "custom"],
                       help="Configuration preset")
    
    # Data paths
    parser.add_argument("--train_path", type=str, default=None,
                       help="Path to training data")
    parser.add_argument("--val_path", type=str, default=None,
                       help="Path to validation data")
    parser.add_argument("--tokenizer_path", type=str, default=None,
                       help="Path to tokenizer")
    
    # Model
    parser.add_argument("--d_model", type=int, default=None)
    parser.add_argument("--n_heads", type=int, default=None)
    parser.add_argument("--n_layers", type=int, default=None)
    parser.add_argument("--n_latent_steps", type=int, default=None)
    parser.add_argument("--n_recursion_steps", type=int, default=None)
    parser.add_argument("--n_supervision_steps", type=int, default=None)
    
    # Training
    parser.add_argument("--batch_size", type=int, default=None)
    parser.add_argument("--learning_rate", type=float, default=None)
    parser.add_argument("--num_epochs", type=int, default=None)
    parser.add_argument("--grad_clip", type=float, default=None)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=None)
    
    # EMA & ACT
    parser.add_argument("--use_ema", action="store_true", default=None)
    parser.add_argument("--no_ema", action="store_true")
    parser.add_argument("--use_act", action="store_true", default=None)
    parser.add_argument("--no_act", action="store_true")
    
    # Misc
    parser.add_argument("--output_dir", type=str, default=None)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--resume", type=str, default=None,
                       help="Path to checkpoint to resume from")
    parser.add_argument("--max_seq_len", type=int, default=None)
    parser.add_argument("--num_workers", type=int, default=4)
    
    args = parser.parse_args()
    
    # Load config
    if args.config == "tool_calling":
        config = TOOL_CALLING_CONFIG
    else:
        config = Config()
    
    # Override config with args
    if args.train_path:
        config.data.train_path = args.train_path
    if args.val_path:
        config.data.val_path = args.val_path
    if args.tokenizer_path:
        config.data.tokenizer_path = args.tokenizer_path
    if args.d_model:
        config.model.d_model = args.d_model
    if args.n_heads:
        config.model.n_heads = args.n_heads
    if args.n_layers:
        config.model.n_layers = args.n_layers
    if args.n_latent_steps:
        config.model.n_latent_steps = args.n_latent_steps
    if args.n_recursion_steps:
        config.model.n_recursion_steps = args.n_recursion_steps
    if args.n_supervision_steps:
        config.model.n_supervision_steps = args.n_supervision_steps
    if args.batch_size:
        config.train.batch_size = args.batch_size
    if args.learning_rate:
        config.train.learning_rate = args.learning_rate
    if args.num_epochs:
        config.train.num_epochs = args.num_epochs
    if args.grad_clip:
        config.train.grad_clip = args.grad_clip
    if args.gradient_accumulation_steps:
        config.train.gradient_accumulation_steps = args.gradient_accumulation_steps
    if args.no_ema:
        config.train.use_ema = False
    elif args.use_ema:
        config.train.use_ema = True
    if args.no_act:
        config.train.use_act = False
    elif args.use_act:
        config.train.use_act = True
    if args.output_dir:
        config.train.output_dir = args.output_dir
    if args.max_seq_len:
        config.data.max_seq_len = args.max_seq_len
        config.model.max_seq_len = args.max_seq_len
    
    # Set seed
    set_seed(args.seed)
    
    print("=" * 60)
    print("TRM Tool Calling Training")
    print("=" * 60)
    
    # Check data files exist
    if not os.path.exists(config.data.train_path):
        print(f"ERROR: Training data not found: {config.data.train_path}")
        print("Run: python scripts/format_hermes_data.py first")
        sys.exit(1)
    
    if not os.path.exists(config.data.tokenizer_path):
        print(f"ERROR: Tokenizer not found: {config.data.tokenizer_path}")
        print("Run: python scripts/train_tokenizer.py first")
        sys.exit(1)
    
    # Get vocab size from tokenizer
    vocab_size = get_tokenizer_vocab_size(config.data.tokenizer_path)
    config.model.vocab_size = vocab_size
    print(f"Vocab size: {vocab_size}")
    
    # Print config
    print("\nConfiguration:")
    print(json.dumps(config.to_dict(), indent=2))
    
    # Create dataloaders
    print("\nLoading data...")
    train_loader, val_loader = create_dataloaders(
        train_path=config.data.train_path,
        val_path=config.data.val_path,
        tokenizer_path=config.data.tokenizer_path,
        batch_size=config.train.batch_size,
        max_seq_len=config.data.max_seq_len,
        num_workers=args.num_workers,
        loss_scale_tool_call=config.train.loss_scale_tool_call,
        loss_scale_think=config.train.loss_scale_think,
        loss_scale_response=config.train.loss_scale_response,
    )
    
    print(f"Train samples: {len(train_loader.dataset)}")
    print(f"Val samples: {len(val_loader.dataset)}")
    
    # Create model
    print("\nCreating model...")
    model = TRMSingleZ(
        vocab_size=config.model.vocab_size,
        d_model=config.model.d_model,
        n_heads=config.model.n_heads,
        n_layers=config.model.n_layers,
        n_latent_steps=config.model.n_latent_steps,
        n_recursion_steps=config.model.n_recursion_steps,
        n_supervision_steps=config.model.n_supervision_steps,
        max_seq_len=config.model.max_seq_len,
        dropout=config.model.dropout,
        use_rotary=config.model.use_rotary,
        use_swiglu=config.model.use_swiglu,
        use_rmsnorm=config.model.use_rmsnorm,
    )
    
    print(f"Model parameters: {model.get_num_params():,}")
    print(f"Effective depth per supervision step: {config.model.effective_depth}")
    
    # Create trainer
    print("\nCreating trainer...")
    trainer = TRMToolTrainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        learning_rate=config.train.learning_rate,
        embedding_lr=config.train.embedding_lr,
        weight_decay=config.train.weight_decay,
        warmup_steps=config.train.warmup_steps,
        grad_clip=config.train.grad_clip,
        gradient_accumulation_steps=config.train.gradient_accumulation_steps,
        n_supervision_steps=config.model.n_supervision_steps,
        use_act=config.train.use_act,
        act_weight=config.train.act_weight,
        use_ema=config.train.use_ema,
        ema_decay=config.train.ema_decay,
        num_epochs=config.train.num_epochs,
        log_interval=config.train.log_interval,
        eval_interval=config.train.eval_interval,
        save_interval=config.train.save_interval,
        output_dir=config.train.output_dir,
    )
    
    # Resume if specified
    if args.resume:
        trainer.load_checkpoint(args.resume)
    
    # Save config
    os.makedirs(config.train.output_dir, exist_ok=True)
    with open(os.path.join(config.train.output_dir, "config.json"), "w") as f:
        json.dump(config.to_dict(), f, indent=2)
    
    # Train
    print("\nStarting training...")
    history = trainer.train()
    
    print("\nTraining complete!")
    print(f"Best validation loss: {trainer.best_val_loss:.4f}")
    print(f"Checkpoints saved to: {config.train.output_dir}")

if __name__ == "__main__":
    main()
