"""
Quick Training Script with Simple Tokenizer
No SentencePiece - trains instantly!
"""
import sys
from pathlib import Path
import argparse
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm

sys.path.append(str(Path(__file__).parent.parent / 'src'))

from simple_tokenizer import SimpleTokenizer
from dataset import ToolCallDataset
from collator import ToolCallCollator
from model import SimpleTRMToolCalling


def train_epoch(model, dataloader, optimizer, device, epoch):
    """Train for one epoch"""
    model.train()
    total_loss = 0
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
        outputs_per_step = model(input_ids, attention_mask, target_args_ids, training=True)
        
        # Compute loss (simplified)
        loss = 0
        for outputs in outputs_per_step:
            tool_logits = outputs['tool_logits']
            tool_loss = nn.functional.cross_entropy(tool_logits, target_tool_id)
            loss += tool_loss
            
            if 'args_logits' in outputs:
                args_logits = outputs['args_logits']
                args_loss = nn.functional.cross_entropy(
                    args_logits.view(-1, args_logits.size(-1)),
                    target_args_ids.view(-1),
                    ignore_index=0,
                )
                loss += args_loss
        
        loss = loss / len(outputs_per_step)
        
        # Backward
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        
        # Stats
        total_loss += loss.item()
        with torch.no_grad():
            predicted = outputs_per_step[-1]['tool_logits'].argmax(dim=-1)
            correct_tools += (predicted == target_tool_id).sum().item()
            total_samples += target_tool_id.size(0)
        
        pbar.set_postfix({'loss': f'{loss.item():.4f}', 'acc': f'{correct_tools/total_samples:.4f}'})
    
    return total_loss / len(dataloader), correct_tools / total_samples


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--train_data', default='data/xlam_1k_swift.json')
    parser.add_argument('--batch_size', type=int, default=4)
    parser.add_argument('--num_epochs', type=int, default=1)
    parser.add_argument('--device', default='cuda' if torch.cuda.is_available() else 'cpu')
    args = parser.parse_args()
    
    print("=" * 80)
    print("Quick Training with Simple Tokenizer")
    print("=" * 80)
    print(f"Device: {args.device}")
    print()
    
    # 1. Train tokenizer (FAST - no SentencePiece!)
    print("Step 1: Training tokenizer...")
    tokenizer = SimpleTokenizer(vocab_size=2000)
    tokenizer.train(args.train_data, 'checkpoints')
    print()
    
    # 2. Create dataset
    print("Step 2: Loading dataset...")
    dataset = ToolCallDataset(args.train_data, tokenizer, max_len=256, max_args_len=64)
    print()
    
    # 3. Create dataloader
    collator = ToolCallCollator(pad_token_id=0)
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True, collate_fn=collator)
    print(f"Batches per epoch: {len(loader)}\n")
    
    # 4. Create model (SMALL for quick testing)
    print("Step 3: Creating model...")
    model = SimpleTRMToolCalling(
        vocab_size=tokenizer.vocab_size_actual(),
        num_tools=len(dataset.tool_to_id),
        hidden_dim=128,
        num_layers=2,
        num_heads=4,
        reasoning_dim=64,
        action_dim=32,
        num_recursions=2,
        max_supervision_steps=2,
    ).to(args.device)
    
    num_params = sum(p.numel() for p in model.parameters())
    print(f"Parameters: {num_params:,}\n")
    
    # 5. Train
    print("Step 4: Training...")
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    
    for epoch in range(args.num_epochs):
        loss, acc = train_epoch(model, loader, optimizer, args.device, epoch)
        print(f"\nEpoch {epoch+1}: Loss={loss:.4f}, Accuracy={acc:.4f}")
    
    # 6. Save
    torch.save({
        'model_state_dict': model.state_dict(),
        'tool_to_id': dataset.tool_to_id,
    }, 'checkpoints/quick_model.pt')
    
    print("\n" + "=" * 80)
    print("✅ Training completed!")
    print("Model saved to: checkpoints/quick_model.pt")
    print("=" * 80)


if __name__ == '__main__':
    main()
