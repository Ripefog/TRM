"""
TRM Single-Z Model Variant for Tool Calling.

This implements the single-z variant from the TRM paper (Algorithm 4):
- Uses only z (latent reasoning feature)
- Output head directly produces logits from z
- Simpler than standard y+z variant
- Good for tasks like tool calling where reasoning is key

Architecture:
    Input x → Embedding
           ↓
    ┌──────────────────────────────────┐
    │   Deep Recursion (T times)       │
    │   ┌────────────────────────────┐ │
    │   │  Latent Recursion (n+1)    │ │
    │   │  z = net(x + z)            │ │
    │   └────────────────────────────┘ │
    └──────────────────────────────────┘
           ↓
    Output Head(z) → logits
    Q Head(z) → halting probability
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import Optional, Tuple
from dataclasses import dataclass

# ============================================================================
# Layer Components
# ============================================================================

class RMSNorm(nn.Module):
    """Root Mean Square Layer Normalization (faster than LayerNorm)"""
    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        rms = torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)
        return x * rms * self.weight

class RotaryEmbedding(nn.Module):
    """Rotary Position Embedding (RoPE)"""
    def __init__(self, dim: int, max_seq_len: int = 2048, base: int = 10000):
        super().__init__()
        inv_freq = 1.0 / (base ** (torch.arange(0, dim, 2).float() / dim))
        self.register_buffer("inv_freq", inv_freq)
        self.max_seq_len = max_seq_len
        self._build_cache(max_seq_len)
    
    def _build_cache(self, seq_len: int):
        t = torch.arange(seq_len, device=self.inv_freq.device)
        freqs = torch.einsum("i,j->ij", t, self.inv_freq)
        emb = torch.cat((freqs, freqs), dim=-1)
        self.register_buffer("cos_cache", emb.cos())
        self.register_buffer("sin_cache", emb.sin())
    
    def forward(self, x: torch.Tensor, seq_len: int) -> Tuple[torch.Tensor, torch.Tensor]:
        if seq_len > self.max_seq_len:
            self._build_cache(seq_len)
        return self.cos_cache[:seq_len], self.sin_cache[:seq_len]

def rotate_half(x: torch.Tensor) -> torch.Tensor:
    x1, x2 = x.chunk(2, dim=-1)
    return torch.cat((-x2, x1), dim=-1)

def apply_rotary_pos_emb(q: torch.Tensor, k: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    # q, k: [batch, n_heads, seq_len, head_dim]
    # cos, sin: [seq_len, head_dim]
    cos = cos.unsqueeze(0).unsqueeze(0)  # [1, 1, seq_len, head_dim]
    sin = sin.unsqueeze(0).unsqueeze(0)
    q_embed = (q * cos) + (rotate_half(q) * sin)
    k_embed = (k * cos) + (rotate_half(k) * sin)
    return q_embed, k_embed

class SwiGLU(nn.Module):
    """SwiGLU activation (better than GELU for LLMs)"""
    def __init__(self, d_model: int, expansion_factor: float = 8/3):
        super().__init__()
        hidden_dim = int(d_model * expansion_factor)
        # Make hidden_dim multiple of 64 for efficiency
        hidden_dim = ((hidden_dim + 63) // 64) * 64
        
        self.w1 = nn.Linear(d_model, hidden_dim, bias=False)
        self.w2 = nn.Linear(d_model, hidden_dim, bias=False)
        self.w3 = nn.Linear(hidden_dim, d_model, bias=False)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.w3(F.silu(self.w1(x)) * self.w2(x))

class MultiHeadAttention(nn.Module):
    """Multi-Head Self-Attention with RoPE"""
    def __init__(self, d_model: int, n_heads: int, dropout: float = 0.1, use_rotary: bool = True, max_seq_len: int = 2048):
        super().__init__()
        assert d_model % n_heads == 0
        
        self.d_model = d_model
        self.n_heads = n_heads
        self.head_dim = d_model // n_heads
        
        self.wq = nn.Linear(d_model, d_model, bias=False)
        self.wk = nn.Linear(d_model, d_model, bias=False)
        self.wv = nn.Linear(d_model, d_model, bias=False)
        self.wo = nn.Linear(d_model, d_model, bias=False)
        
        self.dropout = nn.Dropout(dropout)
        
        self.use_rotary = use_rotary
        if use_rotary:
            self.rotary = RotaryEmbedding(self.head_dim, max_seq_len)
    
    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        batch_size, seq_len, _ = x.size()
        
        # Project
        q = self.wq(x).view(batch_size, seq_len, self.n_heads, self.head_dim).transpose(1, 2)
        k = self.wk(x).view(batch_size, seq_len, self.n_heads, self.head_dim).transpose(1, 2)
        v = self.wv(x).view(batch_size, seq_len, self.n_heads, self.head_dim).transpose(1, 2)
        
        # Apply RoPE
        if self.use_rotary:
            cos, sin = self.rotary(x, seq_len)
            q, k = apply_rotary_pos_emb(q, k, cos, sin)
        
        # Attention
        scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.head_dim)
        
        if mask is not None:
            scores = scores + mask
        
        attn = F.softmax(scores, dim=-1)
        attn = self.dropout(attn)
        
        # Output
        out = torch.matmul(attn, v)
        out = out.transpose(1, 2).contiguous().view(batch_size, seq_len, self.d_model)
        out = self.wo(out)
        
        return out

class TransformerBlock(nn.Module):
    """Single Transformer Block with Pre-Norm"""
    def __init__(self, d_model: int, n_heads: int, dropout: float = 0.1, 
                 use_rotary: bool = True, use_swiglu: bool = True, 
                 use_rmsnorm: bool = True, max_seq_len: int = 2048):
        super().__init__()
        
        # Normalization
        NormClass = RMSNorm if use_rmsnorm else nn.LayerNorm
        self.norm1 = NormClass(d_model)
        self.norm2 = NormClass(d_model)
        
        # Attention
        self.attn = MultiHeadAttention(d_model, n_heads, dropout, use_rotary, max_seq_len)
        
        # FFN
        if use_swiglu:
            self.ffn = SwiGLU(d_model)
        else:
            self.ffn = nn.Sequential(
                nn.Linear(d_model, 4 * d_model),
                nn.GELU(),
                nn.Linear(4 * d_model, d_model),
            )
        
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        # Pre-norm attention
        x = x + self.dropout(self.attn(self.norm1(x), mask))
        # Pre-norm FFN
        x = x + self.dropout(self.ffn(self.norm2(x)))
        return x

# ============================================================================
# TRM Single-Z Model
# ============================================================================

class TRMCell(nn.Module):
    """
    The tiny network used for recursion.
    Only 2 layers as per paper - "Less is More"
    """
    def __init__(self, d_model: int, n_heads: int, n_layers: int = 2, 
                 dropout: float = 0.1, use_rotary: bool = True,
                 use_swiglu: bool = True, use_rmsnorm: bool = True,
                 max_seq_len: int = 2048):
        super().__init__()
        
        self.layers = nn.ModuleList([
            TransformerBlock(
                d_model=d_model,
                n_heads=n_heads,
                dropout=dropout,
                use_rotary=use_rotary,
                use_swiglu=use_swiglu,
                use_rmsnorm=use_rmsnorm,
                max_seq_len=max_seq_len
            )
            for _ in range(n_layers)
        ])
    
    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        for layer in self.layers:
            x = layer(x, mask)
        return x

class TRMSingleZ(nn.Module):
    """
    TRM Single-Z variant for Tool Calling.
    
    Algorithm (from paper Figure 4):
    ```
    def latent_recursion(x, z, n=6):
        for i in range(n+1):  # latent recursion
            z = net(x + z)
        return z
    
    def deep_recursion(x, z, n=6, T=3):
        # T-1 times without gradient
        with torch.no_grad():
            for j in range(T-1):
                z = latent_recursion(x, z, n)
        # 1 time with gradient
        z = latent_recursion(x, z, n)
        return z.detach(), output_head(z), Q_head(z)
    ```
    """
    
    def __init__(
        self,
        vocab_size: int,
        d_model: int = 256,
        n_heads: int = 4,
        n_layers: int = 2,
        n_latent_steps: int = 6,  # n in paper
        n_recursion_steps: int = 3,  # T in paper
        n_supervision_steps: int = 16,  # N_sup in paper
        max_seq_len: int = 1024,
        dropout: float = 0.1,
        use_rotary: bool = True,
        use_swiglu: bool = True,
        use_rmsnorm: bool = True,
    ):
        super().__init__()
        
        # Store config
        self.vocab_size = vocab_size
        self.d_model = d_model
        self.n_latent_steps = n_latent_steps
        self.n_recursion_steps = n_recursion_steps
        self.n_supervision_steps = n_supervision_steps
        self.max_seq_len = max_seq_len
        
        # Embeddings
        self.token_embedding = nn.Embedding(vocab_size, d_model)
        self.dropout = nn.Dropout(dropout)
        
        # The single tiny network
        self.net = TRMCell(
            d_model=d_model,
            n_heads=n_heads,
            n_layers=n_layers,
            dropout=dropout,
            use_rotary=use_rotary,
            use_swiglu=use_swiglu,
            use_rmsnorm=use_rmsnorm,
            max_seq_len=max_seq_len,
        )
        
        # Output normalization
        NormClass = RMSNorm if use_rmsnorm else nn.LayerNorm
        self.ln_f = NormClass(d_model)
        
        # Output head (vocab prediction)
        self.output_head = nn.Linear(d_model, vocab_size, bias=False)
        
        # Q-head for ACT (halting probability)
        self.q_head = nn.Sequential(
            nn.Linear(d_model, d_model // 4),
            nn.GELU(),
            nn.Linear(d_model // 4, 1),
        )
        
        # Learnable initialization for z
        self.z_init = nn.Parameter(torch.zeros(1, 1, d_model))
        nn.init.normal_(self.z_init, std=0.02)
        
        # Initialize weights
        self.apply(self._init_weights)
    
    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
    
    def get_num_params(self) -> int:
        """Get total number of parameters"""
        return sum(p.numel() for p in self.parameters())
    
    def latent_recursion(
        self, 
        x: torch.Tensor, 
        z: torch.Tensor, 
        mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Single-Z latent recursion: z = net(x + z) repeated n+1 times
        
        Args:
            x: embedded input [batch, seq, d_model]
            z: latent state [batch, seq, d_model]
            mask: attention mask
        
        Returns:
            Updated z
        """
        for _ in range(self.n_latent_steps + 1):
            combined = x + z
            z = self.net(combined, mask)
        return z
    
    def deep_recursion(
        self,
        x: torch.Tensor,
        z: torch.Tensor,
        mask: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Deep recursion with T loops (T-1 without grad, 1 with grad)
        
        Returns:
            z_detached: detached z for next supervision step
            logits: output logits
            q_hat: halting probability
        """
        # T-1 times without gradient
        with torch.no_grad():
            for _ in range(self.n_recursion_steps - 1):
                z = self.latent_recursion(x, z, mask)
        
        # 1 time with gradient
        z = self.latent_recursion(x, z, mask)
        
        # Compute outputs
        z_normed = self.ln_f(z)
        logits = self.output_head(z_normed)
        q_hat = torch.sigmoid(self.q_head(z_normed)).squeeze(-1)  # [batch, seq]
        
        return z.detach(), logits, q_hat
    
    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        z_init: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Forward pass for ONE supervision step.
        
        Args:
            input_ids: [batch, seq_len]
            attention_mask: [batch, seq_len]
            z_init: Optional initial z from previous supervision step
        
        Returns:
            z_next: z for next supervision step (detached)
            logits: [batch, seq_len, vocab_size]
            q_hat: [batch, seq_len] halting probability
        """
        batch_size, seq_len = input_ids.size()
        device = input_ids.device
        
        # Embed input
        x = self.token_embedding(input_ids)
        x = self.dropout(x)
        
        # Create causal mask
        causal_mask = torch.triu(
            torch.ones(seq_len, seq_len, device=device) * float('-inf'),
            diagonal=1
        )
        
        # Initialize z if not provided
        if z_init is None:
            z_init = self.z_init.expand(batch_size, seq_len, -1)
        
        # Deep recursion
        z_next, logits, q_hat = self.deep_recursion(x, z_init, causal_mask)
        
        return z_next, logits, q_hat
    
    @torch.no_grad()
    def generate(
        self,
        input_ids: torch.Tensor,
        max_new_tokens: int = 256,
        temperature: float = 1.0,
        top_p: float = 0.9,
        n_supervision_steps: int = None,
    ) -> torch.Tensor:
        """
        Generate tokens using full supervision steps.
        """
        if n_supervision_steps is None:
            n_supervision_steps = self.n_supervision_steps
        
        batch_size = input_ids.size(0)
        device = input_ids.device
        
        # Deep supervision loop to improve z
        z = None
        for _ in range(n_supervision_steps):
            z, logits, q_hat = self.forward(input_ids, z_init=z)
        
        # Generate auto-regressively using the final z
        generated = input_ids.clone()
        
        for _ in range(max_new_tokens):
            if generated.size(1) >= self.max_seq_len:
                break
            
            # Get logits for next token
            z, logits, _ = self.forward(generated, z_init=z)
            
            # Sample next token
            next_logits = logits[:, -1, :] / temperature
            
            # Top-p sampling
            sorted_logits, sorted_indices = torch.sort(next_logits, descending=True)
            cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
            sorted_indices_to_remove = cumulative_probs > top_p
            sorted_indices_to_remove[:, 1:] = sorted_indices_to_remove[:, :-1].clone()
            sorted_indices_to_remove[:, 0] = 0
            
            for batch_idx in range(batch_size):
                next_logits[batch_idx, sorted_indices[batch_idx, sorted_indices_to_remove[batch_idx]]] = float('-inf')
            
            probs = F.softmax(next_logits, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1)
            
            generated = torch.cat([generated, next_token], dim=1)
            
            # Check for EOS (assuming id 2)
            if (next_token == 2).all():
                break
        
        return generated

# ============================================================================
# Factory function
# ============================================================================

def create_trm_single_z(config) -> TRMSingleZ:
    """Create TRM Single-Z model from config"""
    return TRMSingleZ(
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

# ============================================================================
# Test
# ============================================================================

if __name__ == "__main__":
    # Test model
    model = TRMSingleZ(
        vocab_size=32000,
        d_model=256,
        n_heads=4,
        n_layers=2,
        n_latent_steps=6,
        n_recursion_steps=3,
        max_seq_len=512,
    )
    
    print(f"Model parameters: {model.get_num_params():,}")
    
    # Test forward
    batch_size, seq_len = 2, 128
    input_ids = torch.randint(0, 32000, (batch_size, seq_len))
    
    z_next, logits, q_hat = model(input_ids)
    
    print(f"z_next shape: {z_next.shape}")
    print(f"logits shape: {logits.shape}")
    print(f"q_hat shape: {q_hat.shape}")
    
    # Calculate effective depth
    n, T, layers = 6, 3, 2
    effective_depth = T * (n + 1) * layers
    print(f"Effective depth per supervision step: {effective_depth}")
