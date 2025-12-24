"""
Unified TRM Model for Tool Calling
Supports all features: Deep Recursion, RMSNorm, SwiGLU
"""
import torch
import torch.nn as nn
from contextlib import nullcontext

from .base_components import create_encoder
from .trm_components import RecursiveReasoningModule, ActionStateModule, OutputHeads


class TRMToolCalling(nn.Module):
    """
    Unified TRM model supporting:
    - Standard / Advanced features
    - Deep Recursion (T parameter)
    - RMSNorm / LayerNorm
    - SwiGLU / Standard FFN
    """
    
    def __init__(
        self,
        vocab_size,
        num_tools,
        hidden_dim=512,
        num_layers=8,
        num_heads=8,
        reasoning_dim=256,
        action_dim=128,
        num_recursions=3,
        max_supervision_steps=6,
        deep_recursion_steps=1,
        dropout=0.1,
        use_swiglu=True,
        use_rmsnorm=True,
    ):
        super().__init__()
        
        # Store config
        self.hidden_dim = hidden_dim
        self.num_recursions = num_recursions
        self.max_supervision_steps = max_supervision_steps
        self.deep_recursion_steps = deep_recursion_steps
        self.use_swiglu = use_swiglu
        self.use_rmsnorm = use_rmsnorm
        
        # Embeddings
        self.token_embedding = nn.Embedding(vocab_size, hidden_dim)
        self.position_embedding = nn.Embedding(2048, hidden_dim)
        self.dropout = nn.Dropout(dropout)
        
        # Encoder
        self.encoder = create_encoder(
            hidden_dim, num_layers, num_heads, dropout, use_swiglu, use_rmsnorm
        )
        
        # TRM components
        self.reasoning_module = RecursiveReasoningModule(
            hidden_dim, reasoning_dim, action_dim, num_recursions, dropout, use_rmsnorm
        )
        self.action_module = ActionStateModule(
            reasoning_dim, action_dim, dropout, use_rmsnorm
        )
        
        # Output heads
        self.output_heads = OutputHeads(action_dim, num_tools, vocab_size, hidden_dim)
        
        # Initial action state
        self.init_y = nn.Parameter(torch.randn(action_dim) * 0.02)
        
        # Initialize weights
        self._init_weights()
    
    def _init_weights(self):
        """Initialize weights"""
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)
    
    def encode(self, input_ids, attention_mask=None):
        """Encode input sequence"""
        batch_size, seq_len = input_ids.shape
        
        # Embeddings
        token_embeds = self.token_embedding(input_ids)
        positions = torch.arange(seq_len, device=input_ids.device).unsqueeze(0)
        pos_embeds = self.position_embedding(positions)
        
        x = self.dropout(token_embeds + pos_embeds)
        
        # Create attention mask
        if attention_mask is not None:
            encoder_mask = (attention_mask == 0)
        else:
            encoder_mask = None
        
        # Encode
        x_encoded = self.encoder(x, src_key_padding_mask=encoder_mask)
        
        return x_encoded
    
    def forward(self, input_ids, attention_mask=None, target_args_ids=None):
        """
        Forward pass with optional Deep Recursion
        
        Args:
            input_ids: (batch_size, seq_len)
            attention_mask: (batch_size, seq_len)
            target_args_ids: (batch_size, args_len) - for teacher forcing
        
        Returns:
            outputs_per_step: List of dicts, one per supervision step
        """
        batch_size = input_ids.size(0)
        
        # Encode
        x_encoded = self.encode(input_ids, attention_mask)
        
        # Initialize states
        y = self.init_y.unsqueeze(0).expand(batch_size, -1)
        z = None
        
        # Deep supervision loop
        outputs_per_step = []
        
        for step in range(self.max_supervision_steps):
            # Deep Recursion (T iterations)
            for t in range(self.deep_recursion_steps):
                # No gradients for first T-1 iterations
                if self.deep_recursion_steps > 1:
                    context = torch.no_grad() if t < self.deep_recursion_steps - 1 else nullcontext()
                else:
                    context = nullcontext()
                
                with context:
                    # Recursive reasoning
                    z_new = self.reasoning_module(x_encoded, y, z, self.num_recursions)
                    
                    # Update action
                    y_new = self.action_module(y, z_new)
                    
                    # Update states for next iteration
                    if t < self.deep_recursion_steps - 1:
                        z = z_new.detach()
                        y = y_new.detach()
                    else:
                        z = z_new
                        y = y_new
            
            # Output heads (after T iterations)
            is_last_step = (step == self.max_supervision_steps - 1)
            outputs = self.output_heads(
                y,
                token_embedding=self.token_embedding if is_last_step else None,
                target_args_ids=target_args_ids if is_last_step else None
            )
            
            outputs_per_step.append(outputs)
            
            # Detach for next supervision step
            y = y.detach()
            if z is not None:
                z = z.detach()
        
        return outputs_per_step
    
    def get_config(self):
        """Get model configuration"""
        return {
            'hidden_dim': self.hidden_dim,
            'num_recursions': self.num_recursions,
            'max_supervision_steps': self.max_supervision_steps,
            'deep_recursion_steps': self.deep_recursion_steps,
            'use_swiglu': self.use_swiglu,
            'use_rmsnorm': self.use_rmsnorm,
        }
