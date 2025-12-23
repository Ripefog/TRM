"""
Trainer for TRM Single-Z Tool Calling.

Implements:
- Deep Supervision training loop
- ACT (Adaptive Computation Time) for early stopping
- EMA (Exponential Moving Average) for stability
- Weighted loss for tool calls vs responses
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.cuda.amp import autocast, GradScaler
from typing import Optional, Dict, Any, Tuple
import os
import json
from tqdm import tqdm
from copy import deepcopy
import math

class EMA:
    """
    Exponential Moving Average of model weights.
    Crucial for stability on small datasets (from TRM paper).
    """
    def __init__(self, model: nn.Module, decay: float = 0.999):
        self.model = model
        self.decay = decay
        self.shadow = {}
        self.backup = {}
        
        # Initialize shadow weights
        for name, param in model.named_parameters():
            if param.requires_grad:
                self.shadow[name] = param.data.clone()
    
    def update(self):
        """Update shadow weights"""
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                self.shadow[name] = (
                    self.decay * self.shadow[name] + 
                    (1 - self.decay) * param.data
                )
    
    def apply_shadow(self):
        """Apply shadow weights to model (for evaluation)"""
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                self.backup[name] = param.data.clone()
                param.data = self.shadow[name]
    
    def restore(self):
        """Restore original weights"""
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                param.data = self.backup[name]
        self.backup = {}

class TRMToolTrainer:
    """
    Trainer for TRM Single-Z model on Tool Calling task.
    
    Training loop:
    ```
    for x_input, y_true in dataloader:
        z = z_init
        for step in range(N_sup):  # Deep Supervision
            x = input_embedding(x_input)
            z, y_hat, q_hat = deep_recursion(x, z)
            loss = cross_entropy(y_hat, y_true)
            loss += bce(q_hat, (y_hat == y_true))  # ACT loss
            loss.backward()
            optimizer.step()
            z = z.detach()
            if q_hat > 0:  # Early stopping
                break
    ```
    """
    
    def __init__(
        self,
        model: nn.Module,
        train_loader: DataLoader,
        val_loader: Optional[DataLoader] = None,
        # Optimization
        learning_rate: float = 1e-4,
        embedding_lr: float = 1e-2,
        weight_decay: float = 1.0,
        warmup_steps: int = 2000,
        grad_clip: float = 1.0,
        gradient_accumulation_steps: int = 4,
        # TRM specific
        n_supervision_steps: int = 16,
        use_act: bool = True,
        act_weight: float = 1.0,
        # EMA
        use_ema: bool = True,
        ema_decay: float = 0.999,
        # Training
        num_epochs: int = 60,
        # Logging
        log_interval: int = 10,
        eval_interval: int = 500,
        save_interval: int = 1000,
        output_dir: str = "checkpoints",
        # Mixed precision
        use_amp: bool = True,
        # Device
        device: str = None,
    ):
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        
        # Hyperparameters
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.warmup_steps = warmup_steps
        self.grad_clip = grad_clip
        self.gradient_accumulation_steps = gradient_accumulation_steps
        self.n_supervision_steps = n_supervision_steps
        self.use_act = use_act
        self.act_weight = act_weight
        self.use_ema = use_ema
        self.num_epochs = num_epochs
        self.log_interval = log_interval
        self.eval_interval = eval_interval
        self.save_interval = save_interval
        self.output_dir = output_dir
        self.use_amp = use_amp and torch.cuda.is_available()
        
        # Device
        if device is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)
        
        self.model.to(self.device)
        
        # Optimizer with different LR for embeddings
        param_groups = [
            {
                "params": [p for n, p in model.named_parameters() 
                          if "embedding" not in n and p.requires_grad],
                "lr": learning_rate,
            },
            {
                "params": [p for n, p in model.named_parameters() 
                          if "embedding" in n and p.requires_grad],
                "lr": embedding_lr,
            },
        ]
        
        self.optimizer = torch.optim.AdamW(
            param_groups,
            lr=learning_rate,
            betas=(0.9, 0.95),
            weight_decay=weight_decay,
        )
        
        # Loss functions
        self.ce_loss = nn.CrossEntropyLoss(ignore_index=-100, reduction='none')
        self.bce_loss = nn.BCELoss(reduction='none')
        
        # EMA
        if use_ema:
            self.ema = EMA(model, decay=ema_decay)
        else:
            self.ema = None
        
        # Mixed precision
        self.scaler = GradScaler() if self.use_amp else None
        
        # Training state
        self.global_step = 0
        self.best_val_loss = float('inf')
        
        # Create output directory
        os.makedirs(output_dir, exist_ok=True)
    
    def get_lr(self) -> float:
        """Get current learning rate with warmup"""
        if self.global_step < self.warmup_steps:
            return self.learning_rate * self.global_step / self.warmup_steps
        return self.learning_rate
    
    def update_lr(self):
        """Update learning rate"""
        lr = self.get_lr()
        for i, param_group in enumerate(self.optimizer.param_groups):
            if i == 0:  # Main params
                param_group['lr'] = lr
            else:  # Embeddings
                param_group['lr'] = lr * 100  # embedding_lr is 100x main lr
    
    def compute_loss(
        self,
        logits: torch.Tensor,
        labels: torch.Tensor,
        loss_weights: torch.Tensor,
        q_hat: torch.Tensor,
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        """
        Compute total loss with optional weighting and ACT.
        
        Args:
            logits: [batch, seq, vocab]
            labels: [batch, seq]
            loss_weights: [batch, seq] - weights for different parts
            q_hat: [batch, seq] - halting probability
        
        Returns:
            Total loss and metrics dict
        """
        batch_size, seq_len, vocab_size = logits.shape
        
        # CE Loss
        logits_flat = logits.view(-1, vocab_size)
        labels_flat = labels.view(-1)
        ce_loss_flat = self.ce_loss(logits_flat, labels_flat)  # [batch * seq]
        
        # Apply loss weights
        loss_weights_flat = loss_weights.view(-1)
        weighted_ce = ce_loss_flat * loss_weights_flat
        
        # Mask for valid tokens
        valid_mask = (labels_flat != -100).float()
        
        # Mean over valid tokens
        ce_loss = weighted_ce.sum() / (valid_mask.sum() + 1e-8)
        
        metrics = {"ce_loss": ce_loss.item()}
        
        total_loss = ce_loss
        
        # ACT Loss
        if self.use_act:
            # Target: 1 if prediction is correct, 0 otherwise
            with torch.no_grad():
                preds = torch.argmax(logits, dim=-1)  # [batch, seq]
                correct = (preds == labels).float()
                # Only consider valid positions
                valid_mask_2d = (labels != -100).float()
            
            # BCE loss for halting
            if valid_mask_2d.sum() > 0:
                q_hat_valid = q_hat[valid_mask_2d.bool()]
                target_valid = correct[valid_mask_2d.bool()]
                act_loss = self.bce_loss(q_hat_valid, target_valid).mean()
                total_loss = total_loss + self.act_weight * act_loss
                metrics["act_loss"] = act_loss.item()
        
        # Accuracy
        with torch.no_grad():
            preds = torch.argmax(logits, dim=-1)
            valid_mask_2d = labels != -100
            correct = (preds == labels) & valid_mask_2d
            accuracy = correct.float().sum() / (valid_mask_2d.float().sum() + 1e-8)
            metrics["accuracy"] = accuracy.item()
        
        return total_loss, metrics
    
    def train_step(self, batch: Dict[str, torch.Tensor]) -> Dict[str, float]:
        """
        Single training step with Deep Supervision.
        """
        self.model.train()
        
        input_ids = batch["input_ids"].to(self.device)
        labels = batch["labels"].to(self.device)
        loss_weights = batch["loss_weights"].to(self.device)
        
        batch_metrics = {
            "loss": 0.0,
            "ce_loss": 0.0,
            "act_loss": 0.0,
            "accuracy": 0.0,
            "supervision_steps": 0,
        }
        
        z = None  # Start with learnable init
        
        # Deep Supervision Loop
        for step in range(self.n_supervision_steps):
            # Forward pass
            if self.use_amp:
                with autocast():
                    z_next, logits, q_hat = self.model(input_ids, z_init=z)
                    loss, metrics = self.compute_loss(logits, labels, loss_weights, q_hat)
                    loss = loss / self.gradient_accumulation_steps
            else:
                z_next, logits, q_hat = self.model(input_ids, z_init=z)
                loss, metrics = self.compute_loss(logits, labels, loss_weights, q_hat)
                loss = loss / self.gradient_accumulation_steps
            
            # Backward
            if self.use_amp:
                self.scaler.scale(loss).backward()
            else:
                loss.backward()
            
            # Accumulate metrics
            batch_metrics["loss"] += loss.item() * self.gradient_accumulation_steps
            for k, v in metrics.items():
                batch_metrics[k] += v
            batch_metrics["supervision_steps"] += 1
            
            # Detach z for next step
            z = z_next
            
            # Early stopping with ACT
            if self.use_act:
                # Check if model is confident (q > 0.5 on average)
                with torch.no_grad():
                    mean_q = q_hat.mean().item()
                    if mean_q > 0.5 and step > 0:
                        break
        
        # Average metrics over supervision steps
        n_steps = batch_metrics["supervision_steps"]
        for k in ["ce_loss", "act_loss", "accuracy"]:
            batch_metrics[k] /= n_steps
        
        return batch_metrics
    
    def train(self):
        """Main training loop"""
        print(f"Starting training on {self.device}")
        print(f"Model parameters: {self.model.get_num_params():,}")
        print(f"Epochs: {self.num_epochs}")
        print(f"Batch size: {self.train_loader.batch_size}")
        print(f"Gradient accumulation: {self.gradient_accumulation_steps}")
        print(f"Supervision steps: {self.n_supervision_steps}")
        
        history = []
        accumulation_counter = 0
        
        for epoch in range(self.num_epochs):
            epoch_metrics = {
                "loss": 0.0,
                "ce_loss": 0.0,
                "accuracy": 0.0,
                "supervision_steps": 0.0,
            }
            
            pbar = tqdm(self.train_loader, desc=f"Epoch {epoch+1}/{self.num_epochs}")
            
            for batch_idx, batch in enumerate(pbar):
                # Update learning rate
                self.update_lr()
                
                # Training step
                metrics = self.train_step(batch)
                
                # Accumulate gradients
                accumulation_counter += 1
                
                if accumulation_counter >= self.gradient_accumulation_steps:
                    # Gradient clipping
                    if self.use_amp:
                        self.scaler.unscale_(self.optimizer)
                    
                    torch.nn.utils.clip_grad_norm_(
                        self.model.parameters(), 
                        self.grad_clip
                    )
                    
                    # Optimizer step
                    if self.use_amp:
                        self.scaler.step(self.optimizer)
                        self.scaler.update()
                    else:
                        self.optimizer.step()
                    
                    self.optimizer.zero_grad()
                    accumulation_counter = 0
                    
                    # Update EMA
                    if self.ema:
                        self.ema.update()
                    
                    self.global_step += 1
                
                # Update metrics
                for k, v in metrics.items():
                    if k in epoch_metrics:
                        epoch_metrics[k] += v
                
                # Update progress bar
                pbar.set_postfix({
                    "loss": f"{metrics['loss']:.4f}",
                    "acc": f"{metrics['accuracy']:.4f}",
                    "steps": metrics['supervision_steps'],
                    "lr": f"{self.get_lr():.2e}",
                })
                
                # Evaluation
                if self.global_step > 0 and self.global_step % self.eval_interval == 0:
                    if self.val_loader:
                        val_loss = self.evaluate()
                        print(f"\nStep {self.global_step} - Val Loss: {val_loss:.4f}")
                        
                        if val_loss < self.best_val_loss:
                            self.best_val_loss = val_loss
                            self.save_checkpoint("best")
                
                # Save checkpoint
                if self.global_step > 0 and self.global_step % self.save_interval == 0:
                    self.save_checkpoint(f"step_{self.global_step}")
            
            # Epoch summary
            n_batches = len(self.train_loader)
            for k in epoch_metrics:
                epoch_metrics[k] /= n_batches
            
            print(f"\nEpoch {epoch+1} Summary:")
            print(f"  Loss: {epoch_metrics['loss']:.4f}")
            print(f"  Accuracy: {epoch_metrics['accuracy']:.4f}")
            print(f"  Avg Supervision Steps: {epoch_metrics['supervision_steps']:.2f}")
            
            history.append({
                "epoch": epoch + 1,
                **epoch_metrics,
            })
            
            # Save epoch checkpoint
            self.save_checkpoint(f"epoch_{epoch+1}")
        
        # Save final checkpoint
        self.save_checkpoint("final")
        
        # Save history
        with open(os.path.join(self.output_dir, "history.json"), "w") as f:
            json.dump(history, f, indent=2)
        
        print("Training complete!")
        return history
    
    @torch.no_grad()
    def evaluate(self) -> float:
        """Evaluate on validation set"""
        if self.ema:
            self.ema.apply_shadow()
        
        self.model.eval()
        total_loss = 0.0
        total_samples = 0
        
        for batch in tqdm(self.val_loader, desc="Evaluating"):
            input_ids = batch["input_ids"].to(self.device)
            labels = batch["labels"].to(self.device)
            loss_weights = batch["loss_weights"].to(self.device)
            
            z = None
            
            # Full supervision steps for evaluation
            for _ in range(self.n_supervision_steps):
                z, logits, q_hat = self.model(input_ids, z_init=z)
            
            # Compute loss
            loss, _ = self.compute_loss(logits, labels, loss_weights, q_hat)
            
            total_loss += loss.item() * input_ids.size(0)
            total_samples += input_ids.size(0)
        
        if self.ema:
            self.ema.restore()
        
        self.model.train()
        return total_loss / total_samples
    
    def save_checkpoint(self, name: str):
        """Save model checkpoint"""
        checkpoint = {
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "global_step": self.global_step,
            "best_val_loss": self.best_val_loss,
        }
        
        if self.ema:
            checkpoint["ema_shadow"] = self.ema.shadow
        
        if self.scaler:
            checkpoint["scaler_state_dict"] = self.scaler.state_dict()
        
        path = os.path.join(self.output_dir, f"checkpoint_{name}.pt")
        torch.save(checkpoint, path)
        print(f"Saved checkpoint: {path}")
    
    def load_checkpoint(self, path: str):
        """Load model checkpoint"""
        checkpoint = torch.load(path, map_location=self.device)
        
        self.model.load_state_dict(checkpoint["model_state_dict"])
        self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        self.global_step = checkpoint["global_step"]
        self.best_val_loss = checkpoint.get("best_val_loss", float('inf'))
        
        if self.ema and "ema_shadow" in checkpoint:
            self.ema.shadow = checkpoint["ema_shadow"]
        
        if self.scaler and "scaler_state_dict" in checkpoint:
            self.scaler.load_state_dict(checkpoint["scaler_state_dict"])
        
        print(f"Loaded checkpoint from {path}")
        print(f"Resuming from step {self.global_step}")
