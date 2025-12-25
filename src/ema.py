"""
Exponential Moving Average (EMA) for model parameters

Maintains a shadow copy of model parameters that is updated as:
    ema_param = decay * ema_param + (1 - decay) * param

Benefits for TRM:
- Stabilizes recursive reasoning by smoothing parameter updates
- The same reasoning network f(x,y,z) is applied n times recursively
- EMA ensures consistent behavior across recursion depths
- Improves generalization on validation/test sets
"""

import torch
import torch.nn as nn
from typing import Optional


class EMAModel:
    """Exponential Moving Average of model parameters
    
    Usage:
        ema = EMAModel(model, decay=0.9999)
        for batch in dataloader:
            loss = model(batch)
            loss.backward()
            optimizer.step()
            ema.update()  # Update EMA after each step
        
        # For evaluation, use EMA weights
        ema.apply_shadow()
        evaluate(model)
        ema.restore()  # Restore original weights for training
    """
    
    def __init__(
        self,
        model: nn.Module,
        decay: float = 0.9999,
        device: Optional[str] = None,
    ):
        """
        Args:
            model: Model to track (can be DDP-wrapped or raw)
            decay: EMA decay rate (higher = slower updates, more smoothing)
                   Typical values: 0.999, 0.9999, 0.99999
            device: Device for EMA parameters (default: same as model)
        """
        self.decay = decay
        self.device = device
        
        # Get raw model (unwrap DDP if needed)
        if hasattr(model, 'module'):
            self.model = model.module
        else:
            self.model = model
        
        # Create shadow parameters (EMA copy)
        self.shadow = {}
        self.backup = {}  # For storing original params during eval
        
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                self.shadow[name] = param.data.clone()
                if device:
                    self.shadow[name] = self.shadow[name].to(device)
    
    def update(self):
        """Update EMA parameters after optimizer step
        
        Formula: ema_param = decay * ema_param + (1 - decay) * param
        """
        with torch.no_grad():
            for name, param in self.model.named_parameters():
                if param.requires_grad and name in self.shadow:
                    self.shadow[name].mul_(self.decay).add_(
                        param.data, alpha=1 - self.decay
                    )
    
    def apply_shadow(self):
        """Apply EMA weights to model (for evaluation)
        
        Swaps current model parameters with EMA shadow parameters.
        Call restore() after evaluation to get back original weights.
        """
        for name, param in self.model.named_parameters():
            if param.requires_grad and name in self.shadow:
                self.backup[name] = param.data.clone()
                param.data.copy_(self.shadow[name])
    
    def restore(self):
        """Restore original weights after evaluation
        
        Swaps EMA shadow parameters back to original parameters.
        """
        for name, param in self.model.named_parameters():
            if param.requires_grad and name in self.backup:
                param.data.copy_(self.backup[name])
        self.backup = {}
    
    def state_dict(self):
        """Get EMA state dict for saving"""
        return {
            'decay': self.decay,
            'shadow': self.shadow,
        }
    
    def load_state_dict(self, state_dict):
        """Load EMA state dict from checkpoint"""
        self.decay = state_dict['decay']
        self.shadow = state_dict['shadow']
        
        # Move to correct device if specified
        if self.device:
            for name in self.shadow:
                self.shadow[name] = self.shadow[name].to(self.device)


if __name__ == '__main__':
    # Test EMA
    import torch.nn as nn
    
    model = nn.Linear(10, 10)
    ema = EMAModel(model, decay=0.999)
    
    # Simulate training
    for i in range(10):
        # Update model weights
        with torch.no_grad():
            for param in model.parameters():
                param.add_(torch.randn_like(param) * 0.1)
        
        # Update EMA
        ema.update()
    
    print("EMA test passed!")
