"""
Base Components for TRM Model
Reusable building blocks: normalization, FFN, embeddings, transformer
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


# ============================================================================
# Normalization
# ============================================================================

class RMSNorm(nn.Module):
    """Root Mean Square Layer Normalization (faster than LayerNorm)"""
    
    def __init__(self, dim, eps=1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(dim))
        self.eps = eps
    
    def forward(self, x):
        rms = torch.sqrt(x.pow(2).mean(dim=-1, keepdim=True) + self.eps)
        return self.weight * (x / rms)


# ============================================================================
# Feed-Forward Networks
# ============================================================================

class SwiGLU(nn.Module):
    """Swish-Gated Linear Unit (better than standard FFN)"""
    
    def __init__(self, hidden_dim, ff_dim):
        super().__init__()
        self.w1 = nn.Linear(hidden_dim, ff_dim, bias=False)  # Gate
        self.w2 = nn.Linear(hidden_dim, ff_dim, bias=False)  # Value
        self.w3 = nn.Linear(ff_dim, hidden_dim, bias=False)  # Output
    
    def forward(self, x):
        gate = F.silu(self.w1(x))  # Swish activation
        value = self.w2(x)
        return self.w3(gate * value)


class StandardFFN(nn.Module):
    """Standard Feed-Forward Network"""
    
    def __init__(self, hidden_dim, ff_dim, dropout=0.1):
        super().__init__()
        self.linear1 = nn.Linear(hidden_dim, ff_dim)
        self.linear2 = nn.Linear(ff_dim, hidden_dim)
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x):
        x = F.relu(self.linear1(x))
        x = self.dropout(x)
        return self.linear2(x)


# ============================================================================
# Transformer Encoder
# ============================================================================

class TransformerEncoderLayerWithSwiGLU(nn.Module):
    """Transformer Encoder Layer with SwiGLU FFN and configurable normalization"""
    
    def __init__(self, hidden_dim, num_heads, ff_dim, dropout=0.1, use_rmsnorm=True):
        super().__init__()
        self.hidden_dim = hidden_dim
        
        # Normalization
        norm_class = RMSNorm if use_rmsnorm else nn.LayerNorm
        self.norm1 = norm_class(hidden_dim)
        self.norm2 = norm_class(hidden_dim)
        
        # Attention
        self.self_attn = nn.MultiheadAttention(
            hidden_dim, num_heads, dropout=dropout, batch_first=True
        )
        
        # FFN
        self.ffn = SwiGLU(hidden_dim, ff_dim)
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x, src_key_padding_mask=None):
        # Self-attention with pre-norm
        x_norm = self.norm1(x)
        attn_out, _ = self.self_attn(
            x_norm, x_norm, x_norm,
            key_padding_mask=src_key_padding_mask,
            need_weights=False
        )
        x = x + self.dropout(attn_out)
        
        # FFN with pre-norm
        x_norm = self.norm2(x)
        ffn_out = self.ffn(x_norm)
        x = x + self.dropout(ffn_out)
        
        return x


class TransformerEncoderWithSwiGLU(nn.Module):
    """Transformer Encoder using SwiGLU layers"""
    
    def __init__(self, hidden_dim, num_layers, num_heads, ff_dim, dropout=0.1, use_rmsnorm=True):
        super().__init__()
        self.layers = nn.ModuleList([
            TransformerEncoderLayerWithSwiGLU(
                hidden_dim, num_heads, ff_dim, dropout, use_rmsnorm
            )
            for _ in range(num_layers)
        ])
    
    def forward(self, x, src_key_padding_mask=None):
        for layer in self.layers:
            x = layer(x, src_key_padding_mask)
        return x


def create_encoder(hidden_dim, num_layers, num_heads, dropout=0.1, use_swiglu=True, use_rmsnorm=True):
    """
    Factory function to create encoder
    
    Args:
        use_swiglu: If True, use SwiGLU FFN; else use standard FFN
        use_rmsnorm: If True, use RMSNorm; else use LayerNorm
    """
    ff_dim = hidden_dim * 4
    
    if use_swiglu:
        return TransformerEncoderWithSwiGLU(
            hidden_dim, num_layers, num_heads, ff_dim, dropout, use_rmsnorm
        )
    else:
        # Standard PyTorch Transformer
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=num_heads,
            dim_feedforward=ff_dim,
            dropout=dropout,
            batch_first=True,
            norm_first=True,
        )
        return nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
