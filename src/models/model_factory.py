"""
Model Factory
Easy creation of TRM models with different sizes
"""
import sys
from pathlib import Path

# Add parent directory to path for configs
sys.path.append(str(Path(__file__).parent.parent.parent))

from configs.model_configs import get_config
from .trm_model import TRMToolCalling


def create_model(
    model_size,
    vocab_size,
    num_tools,
    **overrides
):
    """
    Factory function to create TRM models
    
    Args:
        model_size: 'tiny', 'small', 'base', or 'large'
        vocab_size: Vocabulary size
        num_tools: Number of tools
        **overrides: Override any config parameters
    
    Returns:
        TRMToolCalling model
    
    Example:
        # Create small model with defaults
        model = create_model('small', vocab_size=8000, num_tools=150)
        
        # Create base model with custom settings
        model = create_model('base', vocab_size=8000, num_tools=150,
                           use_swiglu=False, deep_recursion_steps=5)
    """
    # Get base config
    config = get_config(model_size)
    
    # Apply overrides
    config.update(overrides)
    
    # Create model
    model = TRMToolCalling(
        vocab_size=vocab_size,
        num_tools=num_tools,
        **config
    )
    
    return model


def print_model_info(model):
    """Print model information"""
    num_params = sum(p.numel() for p in model.parameters())
    config = model.get_config()
    
    print(f"Model: {num_params:,} parameters ({num_params/1e6:.1f}M)")
    print(f"Config:")
    for key, value in config.items():
        print(f"  {key}: {value}")
