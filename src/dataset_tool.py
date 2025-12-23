"""
Dataset classes for TRM Tool Calling Training.
Supports Hermes/SWIFT format with proper masking for tool calls.
"""

import torch
from torch.utils.data import Dataset, DataLoader
import json
import os
from typing import Dict, List, Optional, Tuple, Any
from tokenizers import Tokenizer
from dataclasses import dataclass
import random

@dataclass
class SpecialTokens:
    """Special token IDs for tool calling"""
    pad_id: int
    bos_id: int
    eos_id: int
    im_start_id: int
    im_end_id: int
    tool_call_start_id: Optional[int] = None
    tool_call_end_id: Optional[int] = None
    think_start_id: Optional[int] = None
    think_end_id: Optional[int] = None

class HermesToolDataset(Dataset):
    """
    Dataset for Hermes/SWIFT format tool calling data.
    
    Expected JSON format:
    [
        {
            "tools": "[{\"type\": \"function\", \"function\": {...}}, ...]",
            "messages": [
                {"role": "user", "content": "..."},
                {"role": "assistant", "content": "<think>...</think>"},
                {"role": "tool_call", "content": "{\"name\": ..., \"arguments\": ...}"},
                {"role": "tool_response", "content": "..."},
                {"role": "assistant", "content": "final answer"}
            ]
        },
        ...
    ]
    
    Masking strategy:
    - System prompt (tools): MASKED (-100)
    - User messages: MASKED (-100)
    - Tool responses: MASKED (-100)
    - Assistant (think): TRAINED (with loss_scale_think)
    - Assistant (tool_call): TRAINED (with loss_scale_tool_call)
    - Assistant (response): TRAINED (with loss_scale_response)
    """
    
    SYSTEM_PROMPT_TEMPLATE = """<|im_start|>system
You are a helpful assistant.

# Tools

You may call one or more functions to assist with the user query.

You are provided with function signatures within <tools></tools> XML tags:
<tools>
{tools}
</tools>

For each function call, return a json object with function name and arguments within <tool_call></tool_call> XML tags:
<tool_call>
{{"name": <function-name>, "arguments": <args-json-object>}}
</tool_call><|im_end|>
"""

    def __init__(
        self,
        data_path: str,
        tokenizer_path: str,
        max_seq_len: int = 1024,
        loss_scale_think: float = 1.0,
        loss_scale_tool_call: float = 2.0,
        loss_scale_response: float = 1.0,
        shuffle_tools: bool = False,
    ):
        self.data_path = data_path
        self.max_seq_len = max_seq_len
        self.loss_scale_think = loss_scale_think
        self.loss_scale_tool_call = loss_scale_tool_call
        self.loss_scale_response = loss_scale_response
        self.shuffle_tools = shuffle_tools
        
        # Load data
        with open(data_path, 'r', encoding='utf-8') as f:
            self.data = json.load(f)
        
        # Load tokenizer
        self.tokenizer = Tokenizer.from_file(tokenizer_path)
        
        # Get special token IDs
        self.special_tokens = self._get_special_tokens()
        
    def _get_special_tokens(self) -> SpecialTokens:
        """Get special token IDs from tokenizer"""
        pad_id = self.tokenizer.token_to_id("<pad>") or 0
        bos_id = self.tokenizer.token_to_id("<s>") or 1
        eos_id = self.tokenizer.token_to_id("</s>") or 2
        
        # ChatML tokens
        im_start_id = self.tokenizer.token_to_id("<|im_start|>")
        im_end_id = self.tokenizer.token_to_id("<|im_end|>")
        
        # Fallback
        if im_start_id is None:
            im_start_id = bos_id
        if im_end_id is None:
            im_end_id = eos_id
            
        return SpecialTokens(
            pad_id=pad_id,
            bos_id=bos_id,
            eos_id=eos_id,
            im_start_id=im_start_id,
            im_end_id=im_end_id,
        )
    
    def __len__(self) -> int:
        return len(self.data)
    
    def _encode(self, text: str) -> List[int]:
        """Encode text to token IDs"""
        return self.tokenizer.encode(text).ids
    
    def _create_system_prompt(self, tools_str: str) -> str:
        """Create system prompt with tools"""
        # Optionally shuffle tools
        if self.shuffle_tools:
            try:
                tools = json.loads(tools_str)
                random.shuffle(tools)
                tools_str = json.dumps(tools, ensure_ascii=False)
            except:
                pass
        
        return self.SYSTEM_PROMPT_TEMPLATE.format(tools=tools_str)
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        item = self.data[idx]
        tools_str = item['tools']
        messages = item['messages']
        
        all_input_ids = []
        all_labels = []
        all_loss_weights = []  # For weighted loss
        
        # 1. Add BOS
        all_input_ids.append(self.special_tokens.bos_id)
        all_labels.append(-100)
        all_loss_weights.append(0.0)
        
        # 2. System prompt (MASKED)
        system_text = self._create_system_prompt(tools_str)
        system_ids = self._encode(system_text)
        all_input_ids.extend(system_ids)
        all_labels.extend([-100] * len(system_ids))
        all_loss_weights.extend([0.0] * len(system_ids))
        
        # 3. Process messages
        for msg in messages:
            role = msg['role']
            content = msg['content']
            
            if role == 'user':
                # User message: MASKED
                text = f"<|im_start|>user\n{content}<|im_end|>\n"
                ids = self._encode(text)
                all_input_ids.extend(ids)
                all_labels.extend([-100] * len(ids))
                all_loss_weights.extend([0.0] * len(ids))
                
            elif role == 'assistant':
                # Assistant message: TRAINED
                # Check if it's a thinking block
                is_think = content.strip().startswith('<think>')
                
                # Header: MASKED
                header = "<|im_start|>assistant\n"
                header_ids = self._encode(header)
                all_input_ids.extend(header_ids)
                all_labels.extend([-100] * len(header_ids))
                all_loss_weights.extend([0.0] * len(header_ids))
                
                # Content: TRAINED
                content_with_end = f"{content}<|im_end|>\n"
                content_ids = self._encode(content_with_end)
                all_input_ids.extend(content_ids)
                all_labels.extend(content_ids)  # Train on this
                
                # Loss weight based on type
                weight = self.loss_scale_think if is_think else self.loss_scale_response
                all_loss_weights.extend([weight] * len(content_ids))
                
            elif role == 'tool_call':
                # Tool call: TRAINED with higher weight
                # Header
                header = "<|im_start|>assistant\n"
                header_ids = self._encode(header)
                all_input_ids.extend(header_ids)
                all_labels.extend([-100] * len(header_ids))
                all_loss_weights.extend([0.0] * len(header_ids))
                
                # Tool call content
                content_text = f"<tool_call>\n{content}\n</tool_call><|im_end|>\n"
                content_ids = self._encode(content_text)
                all_input_ids.extend(content_ids)
                all_labels.extend(content_ids)  # Train on this
                all_loss_weights.extend([self.loss_scale_tool_call] * len(content_ids))
                
            elif role in ['tool_response', 'tool']:
                # Tool response: MASKED (like user input)
                text = f"<|im_start|>user\n<tool_response>\n{content}\n</tool_response><|im_end|>\n"
                ids = self._encode(text)
                all_input_ids.extend(ids)
                all_labels.extend([-100] * len(ids))
                all_loss_weights.extend([0.0] * len(ids))
        
        # 4. Add EOS
        all_input_ids.append(self.special_tokens.eos_id)
        all_labels.append(self.special_tokens.eos_id)
        all_loss_weights.append(1.0)
        
        # 5. Truncate and Pad
        return self._finalize(all_input_ids, all_labels, all_loss_weights)
    
    def _finalize(
        self, 
        input_ids: List[int], 
        labels: List[int],
        loss_weights: List[float]
    ) -> Dict[str, torch.Tensor]:
        """Truncate to max_seq_len and pad"""
        # Truncate
        if len(input_ids) > self.max_seq_len:
            input_ids = input_ids[:self.max_seq_len]
            labels = labels[:self.max_seq_len]
            loss_weights = loss_weights[:self.max_seq_len]
        
        # Create attention mask
        seq_len = len(input_ids)
        attention_mask = [1] * seq_len
        
        # Pad
        padding_len = self.max_seq_len - seq_len
        if padding_len > 0:
            input_ids = input_ids + [self.special_tokens.pad_id] * padding_len
            labels = labels + [-100] * padding_len
            loss_weights = loss_weights + [0.0] * padding_len
            attention_mask = attention_mask + [0] * padding_len
        
        return {
            "input_ids": torch.tensor(input_ids, dtype=torch.long),
            "attention_mask": torch.tensor(attention_mask, dtype=torch.long),
            "labels": torch.tensor(labels, dtype=torch.long),
            "loss_weights": torch.tensor(loss_weights, dtype=torch.float),
        }

class ToolCallCollator:
    """Custom collator for tool calling data"""
    
    def __init__(self, pad_id: int = 0):
        self.pad_id = pad_id
    
    def __call__(self, batch: List[Dict[str, torch.Tensor]]) -> Dict[str, torch.Tensor]:
        return {
            "input_ids": torch.stack([x["input_ids"] for x in batch]),
            "attention_mask": torch.stack([x["attention_mask"] for x in batch]),
            "labels": torch.stack([x["labels"] for x in batch]),
            "loss_weights": torch.stack([x["loss_weights"] for x in batch]),
        }

def create_dataloaders(
    train_path: str,
    val_path: str,
    tokenizer_path: str,
    batch_size: int = 8,
    max_seq_len: int = 1024,
    num_workers: int = 4,
    **dataset_kwargs
) -> Tuple[DataLoader, DataLoader]:
    """Create train and validation dataloaders"""
    
    train_dataset = HermesToolDataset(
        data_path=train_path,
        tokenizer_path=tokenizer_path,
        max_seq_len=max_seq_len,
        shuffle_tools=True,  # Augmentation for training
        **dataset_kwargs
    )
    
    val_dataset = HermesToolDataset(
        data_path=val_path,
        tokenizer_path=tokenizer_path,
        max_seq_len=max_seq_len,
        shuffle_tools=False,  # No augmentation for validation
        **dataset_kwargs
    )
    
    pad_id = train_dataset.special_tokens.pad_id
    collator = ToolCallCollator(pad_id=pad_id)
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        collate_fn=collator,
        pin_memory=True,
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        collate_fn=collator,
        pin_memory=True,
    )
    
    return train_loader, val_loader

# Test
if __name__ == "__main__":
    # Test with sample data
    test_data = [
        {
            "tools": '[{"type": "function", "function": {"name": "get_weather", "description": "Get weather info", "parameters": {"type": "object", "properties": {"city": {"type": "string"}}, "required": ["city"]}}}]',
            "messages": [
                {"role": "user", "content": "What's the weather in Tokyo?"},
                {"role": "assistant", "content": "<think>\nUser wants weather info. I should use get_weather.\n</think>"},
                {"role": "tool_call", "content": '{"name": "get_weather", "arguments": {"city": "Tokyo"}}'},
                {"role": "tool_response", "content": '{"temperature": 22, "condition": "sunny"}'},
                {"role": "assistant", "content": "The weather in Tokyo is sunny with a temperature of 22°C."}
            ]
        }
    ]
    
    # Save test data
    os.makedirs("data/test", exist_ok=True)
    with open("data/test/sample.json", "w") as f:
        json.dump(test_data, f, indent=2)
    
    print("Test data saved. Run with actual tokenizer to test.")
