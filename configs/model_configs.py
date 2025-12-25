"""
Model Configurations for TRM Tool Calling

Small: ~50M parameters
Base: ~150M parameters
"""

CONFIGS = {
    'tiny': {
        # For quick testing (~800K params)
        'hidden_dim': 128,
        'num_layers': 2,
        'num_heads': 4,
        'reasoning_dim': 64,
        'action_dim': 32,
        'num_recursions': 2,
        'max_supervision_steps': 2,
        'deep_recursion_steps': 1,  # No deep recursion for tiny
        'use_original_trm_grad': False,  # Standard gradient flow
        'q_threshold': 0.8,  # ACT early stopping threshold
        'dropout': 0.1,
        'use_swiglu': False,  # Standard FFN for tiny
        'use_rmsnorm': False,  # LayerNorm for tiny
    },
    
    'small': {
        # ~50M parameters
        'hidden_dim': 512,
        'num_layers': 8,
        'num_heads': 8,
        'reasoning_dim': 256,
        'action_dim': 128,
        'num_recursions': 3,
        'max_supervision_steps': 6,
        'deep_recursion_steps': 3,  # T=3 (2 no-grad + 1 with-grad)
        'use_original_trm_grad': True,  # Use original TRM gradient flow
        'q_threshold': 0.8,  # ACT early stopping threshold
        'dropout': 0.1,
        'use_swiglu': True,  # Use SwiGLU
        'use_rmsnorm': True,  # Use RMSNorm
    },
    
    'base': {
        # ~150M parameters (matches trm_llm default)
        'hidden_dim': 768,
        'num_layers': 12,
        'num_heads': 12,
        'reasoning_dim': 512,
        'action_dim': 256,
        'num_recursions': 3,
        'max_supervision_steps': 8,
        'deep_recursion_steps': 3,  # T=3
        'use_original_trm_grad': True,  # Use original TRM gradient flow
        'q_threshold': 0.8,  # ACT early stopping threshold
        'dropout': 0.1,
        'use_swiglu': True,
        'use_rmsnorm': True,
    },
    
    'large': {
        # ~350M parameters (future)
        'hidden_dim': 1024,
        'num_layers': 24,
        'num_heads': 16,
        'reasoning_dim': 768,
        'action_dim': 512,
        'num_recursions': 3,
        'max_supervision_steps': 8,
        'deep_recursion_steps': 3,  # T=3
        'use_original_trm_grad': True,  # Use original TRM gradient flow
        'q_threshold': 0.8,  # ACT early stopping threshold
        'dropout': 0.1,
        'use_swiglu': True,
        'use_rmsnorm': True,
    },
}


def get_config(model_size='small'):
    """Get model configuration by size name"""
    if model_size not in CONFIGS:
        raise ValueError(f"Unknown model size: {model_size}. Choose from {list(CONFIGS.keys())}")
    return CONFIGS[model_size]


def estimate_parameters(config, vocab_size=8000, num_tools=150):
    """
    Estimate number of parameters for a given config
    
    Args:
        config: Model config dict
        vocab_size: Vocabulary size
        num_tools: Number of tools
    
    Returns:
        dict with parameter breakdown
    """
    h = config['hidden_dim']
    L = config['num_layers']
    r = config['reasoning_dim']
    a = config['action_dim']
    
    # Embeddings
    token_embed = vocab_size * h
    pos_embed = 2048 * h  # max_seq_len
    
    # Encoder (Transformer)
    # Each layer: QKV proj (3*h*h) + O proj (h*h) + FFN (2*h*4h)
    encoder_per_layer = 4 * h * h + 2 * h * (4 * h)
    encoder_total = L * encoder_per_layer
    
    # Reasoning module
    # Pool (h -> r) + reasoning_net (2 layers)
    reasoning = h * r + 2 * ((r + a) * r + r * r)
    
    # Action module
    # Project (r -> a) + update (2 layers)
    action = r * a + 2 * (2 * a * a + a * a)
    
    # Output heads
    tool_head = a * num_tools
    q_head = a * 1
    
    # Generation head
    # args_proj (h -> a) + decoder (2 layers) + gen_proj (a -> vocab)
    args_proj = h * a
    gen_decoder_per_layer = 4 * a * a + 2 * a * (4 * a)
    gen_decoder = 2 * gen_decoder_per_layer
    gen_proj = a * vocab_size
    generation = args_proj + gen_decoder + gen_proj
    
    total = (token_embed + pos_embed + encoder_total + 
             reasoning + action + tool_head + q_head + generation)
    
    return {
        'embeddings_M': (token_embed + pos_embed) / 1e6,
        'encoder_M': encoder_total / 1e6,
        'reasoning_M': reasoning / 1e6,
        'action_M': action / 1e6,
        'output_heads_M': (tool_head + q_head) / 1e6,
        'generation_M': generation / 1e6,
        'total_M': total / 1e6,
    }


def print_config_comparison():
    """Print comparison of all configs"""
    print("=" * 80)
    print("TRM Model Configurations")
    print("=" * 80)
    
    for name in ['tiny', 'small', 'base', 'large']:
        config = CONFIGS[name]
        params = estimate_parameters(config)
        
        print(f"\n{name.upper()}:")
        print(f"  Architecture: {config['num_layers']}L-{config['hidden_dim']}H-{config['num_heads']}A")
        print(f"  TRM: reasoning={config['reasoning_dim']}, action={config['action_dim']}, "
              f"recursions={config['num_recursions']}, steps={config['max_supervision_steps']}")
        print(f"  Parameters: {params['total_M']:.1f}M")
        print(f"    - Embeddings: {params['embeddings_M']:.1f}M")
        print(f"    - Encoder: {params['encoder_M']:.1f}M")
        print(f"    - Reasoning: {params['reasoning_M']:.1f}M")
        print(f"    - Action: {params['action_M']:.1f}M")
        print(f"    - Output heads: {params['output_heads_M']:.1f}M")
        print(f"    - Generation: {params['generation_M']:.1f}M")
    
    print("\n" + "=" * 80)


if __name__ == '__main__':
    print_config_comparison()
