"""
Metrics computation for TRM model evaluation

Provides comprehensive metrics including:
- Tool selection accuracy
- Macro F1 (for imbalanced datasets)
- Per-class metrics (precision, recall, F1)
- Arguments generation accuracy
"""

import torch
import numpy as np
from typing import Dict, List


def compute_tool_metrics(
    all_predictions: List[int],
    all_targets: List[int],
    num_tools: int
) -> Dict[str, float]:
    """Compute comprehensive tool selection metrics
    
    Args:
        all_predictions: List of predicted tool IDs
        all_targets: List of target tool IDs
        num_tools: Total number of tools
    
    Returns:
        Dictionary with metrics:
        - accuracy: Overall accuracy
        - macro_f1: Macro-averaged F1 (better for imbalanced data)
        - per_class_acc: Per-class accuracy
        - precision, recall, f1 per class
    """
    preds = np.array(all_predictions)
    targets = np.array(all_targets)
    
    # Overall accuracy
    accuracy = (preds == targets).mean()
    
    # Per-class metrics
    per_class_metrics = {}
    f1_scores = []
    
    for tool_id in range(num_tools):
        # Mask for this tool
        target_mask = (targets == tool_id)
        pred_mask = (preds == tool_id)
        
        # True positives, false positives, false negatives
        tp = ((preds == tool_id) & (targets == tool_id)).sum()
        fp = ((preds == tool_id) & (targets != tool_id)).sum()
        fn = ((preds != tool_id) & (targets == tool_id)).sum()
        
        # Precision, Recall, F1
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
        
        # Per-class accuracy (support > 0)
        if target_mask.sum() > 0:
            class_acc = (preds[target_mask] == tool_id).mean()
            per_class_metrics[f'tool_{tool_id}_acc'] = float(class_acc)
            per_class_metrics[f'tool_{tool_id}_precision'] = float(precision)
            per_class_metrics[f'tool_{tool_id}_recall'] = float(recall)
            per_class_metrics[f'tool_{tool_id}_f1'] = float(f1)
            f1_scores.append(f1)
    
    # Macro F1 (average F1 across classes)
    macro_f1 = np.mean(f1_scores) if f1_scores else 0.0
    
    return {
        'accuracy': float(accuracy),
        'macro_f1': float(macro_f1),
        **per_class_metrics
    }


def compute_args_accuracy(
    args_logits: torch.Tensor,
    target_args_ids: torch.Tensor,
    pad_token_id: int = 0
) -> Dict[str, float]:
    """Compute arguments generation accuracy
    
    Args:
        args_logits: (batch_size, seq_len, vocab_size)
        target_args_ids: (batch_size, seq_len)
        pad_token_id: Padding token ID to ignore
    
    Returns:
        Dictionary with:
        - token_accuracy: Per-token accuracy (ignoring padding)
        - sequence_accuracy: Exact match accuracy (entire sequence correct)
    """
    # Get predictions
    preds = args_logits.argmax(dim=-1)  # (batch_size, seq_len)
    
    # Create mask (ignore padding)
    mask = (target_args_ids != pad_token_id)
    
    # Token-level accuracy
    correct_tokens = ((preds == target_args_ids) & mask).sum().item()
    total_tokens = mask.sum().item()
    token_accuracy = correct_tokens / total_tokens if total_tokens > 0 else 0.0
    
    # Sequence-level accuracy (exact match)
    # For each sequence, check if ALL non-padding tokens match
    seq_correct = []
    for i in range(preds.size(0)):
        seq_mask = mask[i]
        if seq_mask.sum() > 0:
            # Check if all valid tokens match
            all_match = ((preds[i] == target_args_ids[i]) | ~seq_mask).all()
            seq_correct.append(all_match.item())
    
    sequence_accuracy = np.mean(seq_correct) if seq_correct else 0.0
    
    return {
        'args_token_accuracy': float(token_accuracy),
        'args_sequence_accuracy': float(sequence_accuracy),
    }


def compute_combined_accuracy(
    tool_correct: bool,
    args_correct: bool
) -> float:
    """Compute combined accuracy (tool AND args both correct)
    
    Args:
        tool_correct: Whether tool selection is correct
        args_correct: Whether args generation is correct
    
    Returns:
        1.0 if both correct, 0.0 otherwise
    """
    return 1.0 if (tool_correct and args_correct) else 0.0
