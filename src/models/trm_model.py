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
        use_original_trm_grad=False,
        q_threshold=0.8,
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
        self.use_original_trm_grad = use_original_trm_grad
        self.q_threshold = q_threshold
        
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
    
    def _latent_recursion(self, x_encoded, y, z, n_recursions):
        """Single latent recursion: refine z n times, then update y
        
        This is the inner loop from TRM paper:
            for i in range(n):
                z = net(x, y, z)
            y = net(y, z)
        
        Args:
            x_encoded: (batch_size, seq_len, hidden_dim) - encoded input
            y: (batch_size, action_dim) - current action state
            z: (batch_size, reasoning_dim) or None - current reasoning state
            n_recursions: number of reasoning refinement iterations
        
        Returns:
            y: updated action state
            z: updated reasoning state
        """
        z = self.reasoning_module(x_encoded, y, z, n_recursions)
        y = self.action_module(y, z)
        return y, z
    
    def _deep_recursion(self, x_encoded, y, z, n_recursions, T, use_original_grad=False):
        """Deep recursion with T iterations of latent_recursion
        
        Original TRM paper gradient flow:
            - T-1 iterations WITHOUT gradients (torch.no_grad)
            - 1 iteration WITH gradients
            - Returns detached y, z
        
        Args:
            x_encoded: encoded input
            y: current action state
            z: current reasoning state
            n_recursions: n parameter (latent reasoning iterations)
            T: T parameter (deep recursion iterations)
            use_original_grad: if True, use original TRM gradient flow
        
        Returns:
            y: updated action state (detached if use_original_grad)
            z: updated reasoning state (detached if use_original_grad)
        """
        if use_original_grad and T > 1:
            # Original TRM: T-1 iterations without gradients
            with torch.no_grad():
                for j in range(T - 1):
                    y, z = self._latent_recursion(x_encoded, y, z, n_recursions)
            
            # Final iteration with gradients
            y, z = self._latent_recursion(x_encoded, y, z, n_recursions)
            
            # Detach before returning (original TRM behavior)
            return y.detach(), z.detach()
        else:
            # Standard: all T iterations with gradients
            for j in range(T):
                y, z = self._latent_recursion(x_encoded, y, z, n_recursions)
            return y, z
    
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
    
    def forward(self, input_ids, attention_mask=None, target_args_ids=None, training=True):
        """
        Forward pass with Deep Recursion and ACT (Adaptive Computation Time)
        
        Args:
            input_ids: (batch_size, seq_len)
            attention_mask: (batch_size, seq_len)
            target_args_ids: (batch_size, args_len) - for teacher forcing
            training: bool - if False, enables ACT early stopping
        
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
        
        # Get TRM parameters
        T = self.deep_recursion_steps
        use_original_grad = self.use_original_trm_grad
        n_recursions = self.num_recursions
        
        for step in range(self.max_supervision_steps):
            # Deep recursion: T iterations of latent_recursion
            y, z = self._deep_recursion(
                x_encoded=x_encoded,
                y=y,
                z=z,
                n_recursions=n_recursions,
                T=T,
                use_original_grad=use_original_grad
            )
            
            # Output heads (after T iterations)
            is_last_step = (step == self.max_supervision_steps - 1)
            outputs = self.output_heads(
                y,
                token_embedding=self.token_embedding if is_last_step else None,
                target_args_ids=target_args_ids if is_last_step else None
            )
            
            outputs_per_step.append(outputs)
            
            # ACT: Check for early stopping (inference only)
            if not training:
                q_prob = torch.sigmoid(outputs['q_logit']).mean().item()
                if q_prob > self.q_threshold:
                    # Generate on early stop (if not already generated)
                    if target_args_ids is not None and 'args_logits' not in outputs:
                        # Re-run output heads with generation enabled
                        outputs = self.output_heads(
                            y,
                            token_embedding=self.token_embedding,
                            target_args_ids=target_args_ids
                        )
                        # Update the last output in list
                        outputs_per_step[-1] = outputs
                    break
            
            # Detach for next supervision step (if not already detached by use_original_grad)
            if not use_original_grad:
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
