"""
Configuration for TRM Tool Calling Training.
Supports both standard TRM (y+z) and Single-Z variant.
"""

from dataclasses import dataclass, field
from typing import Optional, List
from enum import Enum

class TRMVariant(str, Enum):
    """TRM model variants"""
    STANDARD = "standard"  # Uses both y (answer) and z (latent)
    SINGLE_Z = "single_z"  # Uses only z (latent -> answer)

class TaskType(str, Enum):
    """Training task types"""
    TOOL_CALLING = "tool_calling"
    PUZZLE = "puzzle"  # Sudoku, Maze, etc.
    GENERAL = "general"

@dataclass
class ModelConfig:
    """Model architecture configuration"""
    # Core dimensions
    d_model: int = 256
    n_heads: int = 4
    n_layers: int = 2  # Tiny network: only 2 layers!
    
    # TRM recursion parameters
    n_latent_steps: int = 6    # n in paper: z update iterations
    n_recursion_steps: int = 3  # T in paper: deep recursion loops
    n_supervision_steps: int = 16  # N_sup: max supervision steps
    
    # Model variant
    variant: TRMVariant = TRMVariant.SINGLE_Z
    
    # Vocabulary
    vocab_size: int = 32000
    max_seq_len: int = 1024  # Longer for tool calling
    
    # Regularization
    dropout: float = 0.1
    
    # Architecture options
    use_rotary: bool = True  # Rotary Position Embeddings
    use_swiglu: bool = True  # SwiGLU activation (better than GELU)
    use_rmsnorm: bool = True  # RMSNorm instead of LayerNorm
    
    @property
    def effective_depth(self) -> int:
        """Calculate effective depth per supervision step"""
        return self.n_recursion_steps * (self.n_latent_steps + 1) * self.n_layers

@dataclass  
class TrainConfig:
    """Training configuration"""
    # Optimization
    batch_size: int = 8
    gradient_accumulation_steps: int = 4
    learning_rate: float = 1e-4
    embedding_lr: float = 1e-2  # Higher LR for embeddings (from paper)
    weight_decay: float = 1.0  # Heavy weight decay (from paper)
    
    # Schedule
    num_epochs: int = 60  # Long training (paper uses 60k epochs on small data)
    warmup_steps: int = 2000
    
    # Gradient
    grad_clip: float = 1.0
    
    # EMA (Exponential Moving Average) - crucial for stability
    use_ema: bool = True
    ema_decay: float = 0.999
    
    # ACT (Adaptive Computation Time)
    use_act: bool = True
    act_weight: float = 1.0  # Weight for ACT loss
    
    # Loss scaling
    loss_scale_tool_call: float = 2.0  # Higher weight for tool calls
    loss_scale_think: float = 1.0  # Weight for thinking blocks
    loss_scale_response: float = 1.0  # Weight for final response
    
    # Logging & Saving
    log_interval: int = 10
    eval_interval: int = 500
    save_interval: int = 1000
    output_dir: str = "checkpoints"
    
    # Tracking
    project_name: str = "trm-tool-calling"
    run_name: str = "trm-hermes"
    use_wandb: bool = True
    
    # Reproducibility
    seed: int = 42

@dataclass
class DataConfig:
    """Data configuration"""
    # Paths
    train_path: str = "data/processed/hermes_train_10000.json"
    val_path: str = "data/processed/hermes_val_1000.json"
    tokenizer_path: str = "data/processed/tokenizer.json"
    
    # Data format
    task_type: TaskType = TaskType.TOOL_CALLING
    
    # Sequence length
    max_seq_len: int = 1024
    
    # Special tokens for tool calling
    tool_call_start: str = "<tool_call>"
    tool_call_end: str = "</tool_call>"
    tool_response_start: str = "<tool_response>"
    tool_response_end: str = "</tool_response>"
    think_start: str = "<think>"
    think_end: str = "</think>"
    tools_start: str = "<tools>"
    tools_end: str = "</tools>"
    
    # Data augmentation
    use_augmentation: bool = True
    shuffle_tools: bool = True  # Randomly shuffle tool order
    mask_tool_names: bool = False  # Optional: mask tool names for harder learning
    
    # Filtering
    max_tools: int = 10  # Max number of tools per sample
    min_messages: int = 2  # At least user + response
    max_messages: int = 20  # Cap for very long conversations

@dataclass
class Config:
    """Main configuration combining all configs"""
    model: ModelConfig = field(default_factory=ModelConfig)
    train: TrainConfig = field(default_factory=TrainConfig)
    data: DataConfig = field(default_factory=DataConfig)
    
    def __post_init__(self):
        # Sync max_seq_len
        self.model.max_seq_len = self.data.max_seq_len
    
    def to_dict(self) -> dict:
        """Convert to dictionary for logging"""
        return {
            "model": {
                "d_model": self.model.d_model,
                "n_heads": self.model.n_heads,
                "n_layers": self.model.n_layers,
                "n_latent_steps": self.model.n_latent_steps,
                "n_recursion_steps": self.model.n_recursion_steps,
                "n_supervision_steps": self.model.n_supervision_steps,
                "variant": self.model.variant.value,
                "effective_depth": self.model.effective_depth,
                "vocab_size": self.model.vocab_size,
                "max_seq_len": self.model.max_seq_len,
            },
            "train": {
                "batch_size": self.train.batch_size,
                "learning_rate": self.train.learning_rate,
                "num_epochs": self.train.num_epochs,
                "use_ema": self.train.use_ema,
                "use_act": self.train.use_act,
            },
            "data": {
                "train_path": self.data.train_path,
                "task_type": self.data.task_type.value,
            }
        }
    
    @classmethod
    def for_tool_calling(cls, variant: TRMVariant = TRMVariant.SINGLE_Z) -> "Config":
        """Create config optimized for tool calling task"""
        return cls(
            model=ModelConfig(
                d_model=256,
                n_heads=4,
                n_layers=2,
                n_latent_steps=6,
                n_recursion_steps=3,
                n_supervision_steps=16,
                variant=variant,
                max_seq_len=1024,
                use_rotary=True,
                use_swiglu=True,
            ),
            train=TrainConfig(
                batch_size=8,
                learning_rate=1e-4,
                embedding_lr=1e-2,
                weight_decay=1.0,
                num_epochs=60,
                use_ema=True,
                use_act=True,
            ),
            data=DataConfig(
                task_type=TaskType.TOOL_CALLING,
                max_seq_len=1024,
            )
        )
    
    @classmethod
    def for_puzzle(cls, variant: TRMVariant = TRMVariant.STANDARD) -> "Config":
        """Create config optimized for puzzle tasks (Sudoku, Maze)"""
        return cls(
            model=ModelConfig(
                d_model=512,
                n_heads=8,
                n_layers=2,
                n_latent_steps=6,
                n_recursion_steps=3,
                n_supervision_steps=16,
                variant=variant,
                max_seq_len=256,
            ),
            train=TrainConfig(
                batch_size=768,  # Large batch from paper
                learning_rate=1e-4,
                num_epochs=60000,  # Very long training
            ),
            data=DataConfig(
                task_type=TaskType.PUZZLE,
                max_seq_len=256,
            )
        )

# Preset configs
TOOL_CALLING_CONFIG = Config.for_tool_calling(TRMVariant.SINGLE_Z)
PUZZLE_CONFIG = Config.for_puzzle(TRMVariant.STANDARD)
