"""
Training Script with Model Size Selection and LR Scheduler
"""
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import LambdaLR
from tqdm import tqdm
import argparse
from pathlib import Path
import json
import math
import sys

sys.path.append(str(Path(__file__).parent.parent))

from src.tokenizer import ToolCallTokenizer
from src.dataset import ToolCallDataset
from src.collator import ToolCallCollator
from src.models import create_model
from configs.model_configs import get_config, estimate_parameters


def get_cosine_schedule_with_warmup(optimizer, num_warmup_steps, num_training_steps, min_lr_ratio=0.1):
    """
    Create learning rate scheduler with warmup and cosine decay
    
    Args:
        optimizer: PyTorch optimizer
        num_warmup_steps: Number of warmup steps
        num_training_steps: Total number of training steps
        min_lr_ratio: Minimum LR as ratio of initial LR
    """
    def lr_lambda(current_step):
        # Warmup
        if current_step < num_warmup_steps:
            return float(current_step) / float(max(1, num_warmup_steps))
        
        # Cosine decay
        progress = float(current_step - num_warmup_steps) / float(max(1, num_training_steps - num_warmup_steps))
        cosine_decay = 0.5 * (1.0 + math.cos(math.pi * progress))
        return min_lr_ratio + (1.0 - min_lr_ratio) * cosine_decay
    
    return LambdaLR(optimizer, lr_lambda)


def train_epoch(model, dataloader, optimizer, scheduler, device, epoch):
    """Train for one epoch"""
    model.train()
    total_loss = 0
    tool_loss_sum = 0
    q_loss_sum = 0
    args_loss_sum = 0
    correct_tools = 0
    total_samples = 0
    
    pbar = tqdm(dataloader, desc=f"Epoch {epoch+1}")
    
    for batch in pbar:
        # Move to device
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        target_tool_id = batch['target_tool_id'].to(device)
        target_args_ids = batch['target_args_ids'].to(device)
        
        # Forward
        outputs_per_step = model(input_ids, attention_mask, target_args_ids)
        
        # Compute loss
        loss = 0
        batch_tool_loss = 0
        batch_q_loss = 0
        batch_args_loss = 0
        
        for step, outputs in enumerate(outputs_per_step):
            # Tool selection loss
            tool_logits = outputs['tool_logits']
            tool_loss = nn.functional.cross_entropy(tool_logits, target_tool_id)
            loss += tool_loss
            batch_tool_loss += tool_loss.item()
            
            # Q loss (correctness prediction)
            q_logit = outputs['q_logit'].squeeze()
            predicted_tool = tool_logits.argmax(dim=-1)
            is_correct = (predicted_tool == target_tool_id).float()
            q_loss = nn.functional.binary_cross_entropy_with_logits(q_logit, is_correct)
            loss += 0.5 * q_loss
            batch_q_loss += q_loss.item()
            
            # Arguments generation loss (last step only)
            if 'args_logits' in outputs:
                args_logits = outputs['args_logits']
                args_loss = nn.functional.cross_entropy(
                    args_logits.view(-1, args_logits.size(-1)),
                    target_args_ids.view(-1),
                    ignore_index=0,
                )
                loss += 2.0 * args_loss
                batch_args_loss += args_loss.item()
        
        # Average over steps
        loss = loss / len(outputs_per_step)
        
        # Backward
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        scheduler.step()
        
        # Stats
        total_loss += loss.item()
        tool_loss_sum += batch_tool_loss / len(outputs_per_step)
        q_loss_sum += batch_q_loss / len(outputs_per_step)
        args_loss_sum += batch_args_loss / len(outputs_per_step) if batch_args_loss > 0 else 0
        
        with torch.no_grad():
            predicted = outputs_per_step[-1]['tool_logits'].argmax(dim=-1)
            correct_tools += (predicted == target_tool_id).sum().item()
            total_samples += target_tool_id.size(0)
        
        # Update progress bar
        current_lr = scheduler.get_last_lr()[0]
        pbar.set_postfix({
            'loss': f'{loss.item():.4f}',
            'acc': f'{correct_tools/total_samples:.4f}',
            'lr': f'{current_lr:.2e}'
        })
    
    return {
        'loss': total_loss / len(dataloader),
        'tool_loss': tool_loss_sum / len(dataloader),
        'q_loss': q_loss_sum / len(dataloader),
        'args_loss': args_loss_sum / len(dataloader),
        'accuracy': correct_tools / total_samples,
    }


def evaluate(model, dataloader, device):
    """Evaluate model"""
    model.eval()
    total_loss = 0
    correct_tools = 0
    total_samples = 0
    
    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Evaluating"):
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            target_tool_id = batch['target_tool_id'].to(device)
            target_args_ids = batch['target_args_ids'].to(device)
            
            outputs_per_step = model(input_ids, attention_mask, target_args_ids)
            
            # Compute loss (simplified)
            loss = 0
            for outputs in outputs_per_step:
                tool_logits = outputs['tool_logits']
                loss += nn.functional.cross_entropy(tool_logits, target_tool_id)
            loss = loss / len(outputs_per_step)
            
            total_loss += loss.item()
            predicted = outputs_per_step[-1]['tool_logits'].argmax(dim=-1)
            correct_tools += (predicted == target_tool_id).sum().item()
            total_samples += target_tool_id.size(0)
    
    return {
        'loss': total_loss / len(dataloader),
        'accuracy': correct_tools / total_samples,
    }


def main():
    parser = argparse.ArgumentParser(description='Train TRM for Tool Calling')
    
    # Get project root
    project_root = Path(__file__).parent.parent
    
    # Model
    parser.add_argument('--model_size', type=str, default='small', choices=['tiny', 'small', 'base', 'large'],
                        help='Model size (tiny/small/base/large)')
    
    # Data
    parser.add_argument('--train_data', type=str, default=str(project_root / 'data/xlam_1k_swift.json'))
    parser.add_argument('--eval_data', type=str, default=str(project_root / 'data/xlam_val_200_swift.json'))
    
    # Tokenizer
    parser.add_argument('--tokenizer_path', type=str, default=None)
    parser.add_argument('--vocab_size', type=int, default=8000)
    parser.add_argument('--model_type', type=str, default='bpe', choices=['bpe', 'unigram'])
    
    # Training
    parser.add_argument('--batch_size', type=int, default=8)
    parser.add_argument('--num_epochs', type=int, default=50)
    parser.add_argument('--learning_rate', type=float, default=1e-4)
    parser.add_argument('--weight_decay', type=float, default=0.01)
    parser.add_argument('--warmup_ratio', type=float, default=0.1, help='Warmup steps as ratio of total steps')
    parser.add_argument('--min_lr_ratio', type=float, default=0.1, help='Min LR as ratio of initial LR')
    
    # Misc
    parser.add_argument('--save_dir', type=str, default='checkpoints')
    parser.add_argument('--eval_every', type=int, default=5)
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu')
    
    args = parser.parse_args()
    
    # Get model config
    model_config = get_config(args.model_size)
    
    print("=" * 80)
    print(f"TRM Tool Calling Training - {args.model_size.upper()} Model")
    print("=" * 80)
    print(f"Device: {args.device}")
    print(f"Model size: {args.model_size}")
    print(f"  Architecture: {model_config['num_layers']}L-{model_config['hidden_dim']}H-{model_config['num_heads']}A")
    print(f"  TRM: reasoning={model_config['reasoning_dim']}, action={model_config['action_dim']}")
    print()
    
    # Create save directory
    save_dir = Path(args.save_dir) / args.model_size
    save_dir.mkdir(parents=True, exist_ok=True)
    
    # Save config
    with open(save_dir / 'config.json', 'w') as f:
        config_dict = {**vars(args), **model_config}
        json.dump(config_dict, f, indent=2)
    
    # 1. Train/Load tokenizer
    if args.tokenizer_path and Path(args.tokenizer_path).exists():
        print(f"Loading tokenizer from {args.tokenizer_path}...")
        tokenizer = ToolCallTokenizer(args.tokenizer_path)
    else:
        print("Training new tokenizer...")
        tokenizer = ToolCallTokenizer()
        tokenizer.train(args.train_data, str(save_dir), vocab_size=args.vocab_size, model_type=args.model_type)
    print()
    
    # 2. Create datasets
    print("Creating datasets...")
    train_dataset = ToolCallDataset(args.train_data, tokenizer, max_len=512, max_args_len=128)
    eval_dataset = ToolCallDataset(args.eval_data, tokenizer, max_len=512, max_args_len=128)
    print()
    
    # 3. Create dataloaders
    collator = ToolCallCollator(pad_token_id=tokenizer.pad_id())
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, collate_fn=collator, num_workers=0)
    eval_loader = DataLoader(eval_dataset, batch_size=args.batch_size, shuffle=False, collate_fn=collator, num_workers=0)
    
    print(f"Train batches: {len(train_loader)}")
    print(f"Eval batches: {len(eval_loader)}")
    print()
    
    # 4. Create model
    print("Creating model...")
    model = create_model(
        args.model_size,
        vocab_size=tokenizer.vocab_size(),
        num_tools=len(train_dataset.tool_to_id),
    ).to(args.device)
    
    num_params = sum(p.numel() for p in model.parameters())
    params_est = estimate_parameters(model_config, tokenizer.vocab_size(), len(train_dataset.tool_to_id))
    print(f"Model parameters: {num_params:,} ({num_params/1e6:.1f}M)")
    print(f"Estimated: {params_est['total_M']:.1f}M")
    print()
    
    # 5. Create optimizer and scheduler
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
    
    total_steps = len(train_loader) * args.num_epochs
    warmup_steps = int(total_steps * args.warmup_ratio)
    scheduler = get_cosine_schedule_with_warmup(optimizer, warmup_steps, total_steps, args.min_lr_ratio)
    
    print(f"Training setup:")
    print(f"  Total steps: {total_steps}")
    print(f"  Warmup steps: {warmup_steps}")
    print(f"  Initial LR: {args.learning_rate:.2e}")
    print(f"  Min LR: {args.learning_rate * args.min_lr_ratio:.2e}")
    print()
    
    # 6. Training loop
    print("=" * 80)
    print("Training")
    print("=" * 80)
    
    best_eval_acc = 0.0
    
    for epoch in range(args.num_epochs):
        # Train
        train_metrics = train_epoch(model, train_loader, optimizer, scheduler, args.device, epoch)
        
        print(f"\nEpoch {epoch+1}/{args.num_epochs}")
        print(f"  Train Loss: {train_metrics['loss']:.4f}, Acc: {train_metrics['accuracy']:.4f}")
        print(f"    Tool Loss: {train_metrics['tool_loss']:.4f}, Q Loss: {train_metrics['q_loss']:.4f}, "
              f"Args Loss: {train_metrics['args_loss']:.4f}")
        
        # Evaluate
        if (epoch + 1) % args.eval_every == 0:
            eval_metrics = evaluate(model, eval_loader, args.device)
            print(f"  Eval Loss: {eval_metrics['loss']:.4f}, Acc: {eval_metrics['accuracy']:.4f}")
            
            # Save best model
            if eval_metrics['accuracy'] > best_eval_acc:
                best_eval_acc = eval_metrics['accuracy']
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'scheduler_state_dict': scheduler.state_dict(),
                    'train_metrics': train_metrics,
                    'eval_metrics': eval_metrics,
                    'tool_to_id': train_dataset.tool_to_id,
                    'model_config': model_config,
                }, save_dir / 'best_model.pt')
                print(f"  ✓ Saved best model (acc: {eval_metrics['accuracy']:.4f})")
        
        # Save checkpoint
        if (epoch + 1) % 10 == 0:
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                'tool_to_id': train_dataset.tool_to_id,
                'model_config': model_config,
            }, save_dir / f'checkpoint_epoch_{epoch+1}.pt')
        
        print()
    
    print("=" * 80)
    print("Training completed!")
    print(f"Best eval accuracy: {best_eval_acc:.4f}")
    print(f"Models saved to: {save_dir}")
    print("=" * 80)


if __name__ == '__main__':
    main()
