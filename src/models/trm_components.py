"""
TRM-Specific Components
Recursive reasoning, action state, output heads
"""
import torch
import torch.nn as nn
from .base_components import RMSNorm


class RecursiveReasoningModule(nn.Module):
    """Recursive reasoning module with configurable normalization"""
    
    def __init__(self, hidden_dim, reasoning_dim, action_dim, num_recursions, dropout=0.1, use_rmsnorm=True):
        super().__init__()
        self.num_recursions = num_recursions
        self.reasoning_dim = reasoning_dim
        
        # Pool sequence to reasoning dim
        self.pool = nn.Linear(hidden_dim, reasoning_dim)
        
        # Reasoning network
        norm_class = RMSNorm if use_rmsnorm else nn.LayerNorm
        self.reasoning_net = nn.Sequential(
            nn.Linear(reasoning_dim + action_dim, reasoning_dim),
            norm_class(reasoning_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(reasoning_dim, reasoning_dim),
        )
    
    def forward(self, x_encoded, y_current, z_current, n_recursions=None):
        """
        Recursive reasoning
        
        Args:
            x_encoded: (batch_size, seq_len, hidden_dim)
            y_current: (batch_size, action_dim)
            z_current: (batch_size, reasoning_dim) or None
            n_recursions: Override num_recursions if provided
        """
        batch_size = x_encoded.size(0)
        n = n_recursions if n_recursions is not None else self.num_recursions
        
        # Pool sequence (mean pooling)
        x_pooled = x_encoded.mean(dim=1)
        x_proj = self.pool(x_pooled)
        
        # Initialize z
        if z_current is None:
            z = torch.zeros(batch_size, self.reasoning_dim, device=x_encoded.device)
        else:
            z = z_current
        
        # Recursive refinement
        for i in range(n):
            combined = torch.cat([x_proj, y_current], dim=-1)
            z_refined = self.reasoning_net(combined)
            z = z + z_refined  # Residual connection
        
        return z


class ActionStateModule(nn.Module):
    """Action state update module with configurable normalization"""
    
    def __init__(self, reasoning_dim, action_dim, dropout=0.1, use_rmsnorm=True):
        super().__init__()
        self.project = nn.Linear(reasoning_dim, action_dim)
        
        norm_class = RMSNorm if use_rmsnorm else nn.LayerNorm
        self.update = nn.Sequential(
            nn.Linear(action_dim * 2, action_dim),
            norm_class(action_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(action_dim, action_dim),
        )
    
    def forward(self, y_current, z_reasoning):
        """
        Update action state
        
        Args:
            y_current: (batch_size, action_dim)
            z_reasoning: (batch_size, reasoning_dim)
        """
        z_proj = self.project(z_reasoning)
        combined = torch.cat([y_current, z_proj], dim=-1)
        y_update = self.update(combined)
        return y_current + y_update  # Residual connection


class OutputHeads(nn.Module):
    """Output heads for tool selection, correctness, and generation"""
    
    def __init__(self, action_dim, num_tools, vocab_size, hidden_dim):
        super().__init__()
        
        # Tool selection head
        self.tool_head = nn.Linear(action_dim, num_tools)
        
        # Correctness prediction head (Q)
        self.q_head = nn.Linear(action_dim, 1)
        
        # Generation head
        self.args_proj = nn.Linear(hidden_dim, action_dim)
        
        decoder_layer = nn.TransformerDecoderLayer(
            d_model=action_dim,
            nhead=4,
            dim_feedforward=action_dim * 4,
            dropout=0.1,
            batch_first=True,
            norm_first=True,
        )
        self.gen_decoder = nn.TransformerDecoder(decoder_layer, num_layers=2)
        self.gen_proj = nn.Linear(action_dim, vocab_size)
    
    def forward(self, y, token_embedding=None, target_args_ids=None):
        """
        Compute output heads
        
        Args:
            y: (batch_size, action_dim) - action state
            token_embedding: Embedding layer (for generation)
            target_args_ids: (batch_size, args_len) - for teacher forcing
        
        Returns:
            dict with 'tool_logits', 'q_logit', and optionally 'args_logits'
        """
        outputs = {
            'tool_logits': self.tool_head(y),
            'q_logit': self.q_head(y),
        }
        
        # Generation (if target provided)
        if target_args_ids is not None and token_embedding is not None:
            args_embeds = token_embedding(target_args_ids)
            args_embeds = self.args_proj(args_embeds)
            memory = y.unsqueeze(1)
            decoded = self.gen_decoder(args_embeds, memory)
            outputs['args_logits'] = self.gen_proj(decoded)
        
        return outputs
