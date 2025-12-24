"""
Data Collator for batching
"""
import torch


class ToolCallCollator:
    """Collate batch of samples"""
    
    def __init__(self, pad_token_id=0):
        """
        Initialize collator
        
        Args:
            pad_token_id: ID for padding token
        """
        self.pad_token_id = pad_token_id
    
    def __call__(self, batch):
        """
        Collate batch
        
        Args:
            batch: List of dicts from dataset
        
        Returns:
            {
                'input_ids': (batch_size, max_len),
                'attention_mask': (batch_size, max_len),
                'target_tool_id': (batch_size,),
                'target_args_ids': (batch_size, max_args_len),
                'args_mask': (batch_size, max_args_len),
            }
        """
        # Find max lengths
        max_input_len = max(len(item['input_ids']) for item in batch)
        max_args_len = max(len(item['target_args_ids']) for item in batch)
        
        # Pad sequences
        input_ids = []
        attention_masks = []
        target_args_ids = []
        args_masks = []
        
        for item in batch:
            # Pad input
            ids = item['input_ids']
            padding_len = max_input_len - len(ids)
            padded_ids = ids + [self.pad_token_id] * padding_len
            attention_mask = [1] * len(ids) + [0] * padding_len
            
            input_ids.append(padded_ids)
            attention_masks.append(attention_mask)
            
            # Pad arguments
            args_ids = item['target_args_ids']
            args_padding_len = max_args_len - len(args_ids)
            padded_args_ids = args_ids + [self.pad_token_id] * args_padding_len
            args_mask = [1] * len(args_ids) + [0] * args_padding_len
            
            target_args_ids.append(padded_args_ids)
            args_masks.append(args_mask)
        
        return {
            'input_ids': torch.tensor(input_ids, dtype=torch.long),
            'attention_mask': torch.tensor(attention_masks, dtype=torch.long),
            'target_tool_id': torch.tensor([item['target_tool_id'] for item in batch], dtype=torch.long),
            'target_args_ids': torch.tensor(target_args_ids, dtype=torch.long),
            'args_mask': torch.tensor(args_masks, dtype=torch.long),
        }
