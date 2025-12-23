"""
Format Hermes Reasoning Tool Use dataset to SWIFT Agent format for TRM training.

Source: https://huggingface.co/datasets/interstellarninja/hermes_reasoning_tool_use
Target Format: https://swift.readthedocs.io/en/latest/Instruction/Agent-support.html

Dataset has 51K ShareGPT conversations with:
- single_turn: 1 user request → 1 tool call
- multi_turn: Multiple tool calls with user follow-up
- multi_step: ≥2 sequential tool calls after single user turn
- relevance: No tool suitable → assistant refuses
"""

import json
import os
import re
from typing import List, Dict, Any, Optional
from tqdm import tqdm
from datasets import load_dataset

def extract_tools_from_system(system_content: str) -> Optional[str]:
    """
    Extract tools JSON from system prompt.
    Tools are wrapped in <tools></tools> XML tags.
    """
    # Pattern to match tools block
    pattern = r'<tools>\s*([\s\S]*?)\s*</tools>'
    match = re.search(pattern, system_content)
    
    if match:
        tools_str = match.group(1).strip()
        try:
            # Validate it's valid JSON
            tools = json.loads(tools_str)
            # Return as formatted JSON string
            return json.dumps(tools, ensure_ascii=False)
        except json.JSONDecodeError:
            return None
    return None

def extract_tool_calls_from_assistant(content: str) -> List[Dict[str, Any]]:
    """
    Extract tool calls from assistant response.
    Tool calls are wrapped in <tool_call></tool_call> XML tags.
    
    Returns list of tool call dicts: {"name": ..., "arguments": ...}
    """
    pattern = r'<tool_call>\s*([\s\S]*?)\s*</tool_call>'
    matches = re.findall(pattern, content)
    
    tool_calls = []
    for match in matches:
        try:
            tool_call = json.loads(match.strip())
            tool_calls.append(tool_call)
        except json.JSONDecodeError:
            continue
    
    return tool_calls

def extract_think_block(content: str) -> Optional[str]:
    """
    Extract thinking/reasoning block from assistant response.
    Think blocks are wrapped in <think></think> tags.
    """
    pattern = r'<think>\s*([\s\S]*?)\s*</think>'
    match = re.search(pattern, content)
    return match.group(1).strip() if match else None

def remove_think_and_tool_call_tags(content: str) -> str:
    """
    Remove <think> and <tool_call> blocks from content,
    leaving only the final response text.
    """
    # Remove think blocks
    content = re.sub(r'<think>[\s\S]*?</think>\s*', '', content)
    # Remove tool_call blocks
    content = re.sub(r'<tool_call>[\s\S]*?</tool_call>\s*', '', content)
    return content.strip()

def convert_hermes_to_swift(sample: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    """
    Convert a single Hermes sample to SWIFT Agent format.
    
    Hermes format (ShareGPT):
    {
        "conversations": [  # Can be a list or JSON string
            {"from": "system", "value": "..."},
            {"from": "user", "value": "..."},
            {"from": "assistant", "value": "<think>...</think>\n<tool_call>...</tool_call>"},
            {"from": "tool", "value": "..."},  # tool response
            {"from": "assistant", "value": "final answer"}
        ],
        "tools": [...],
        "scenario_category": "single_turn" | "multi_turn" | "multi_step" | "relevance"
    }
    
    SWIFT format:
    {
        "tools": "[{\"type\": \"function\", \"function\": {...}}, ...]",
        "messages": [
            {"role": "user", "content": "..."},
            {"role": "assistant", "content": "<think>...</think>"},  # optional reasoning
            {"role": "tool_call", "content": "{\"name\": ..., \"arguments\": ...}"},
            {"role": "tool_response", "content": "..."},
            {"role": "assistant", "content": "final answer"}
        ]
    }
    """
    # Get conversations - can be list or JSON string
    conversations = sample.get('conversations', [])
    
    # If conversations is a string, parse it as JSON
    if isinstance(conversations, str):
        try:
            conversations = json.loads(conversations)
        except json.JSONDecodeError:
            return None
    
    if not conversations:
        return None
    
    tools_str = None
    messages = []
    
    for conv in conversations:
        # Support both 'from'/'value' (ShareGPT) and 'role'/'content' (OpenAI) formats
        role = conv.get('from') or conv.get('role', '')
        content = conv.get('value') or conv.get('content', '')
        
        if role == 'system':
            # Extract tools from system prompt
            tools_str = extract_tools_from_system(content)
            # Note: We don't add system message to messages list
            # because SWIFT handles tools separately
            
        elif role == 'user':
            messages.append({
                "role": "user",
                "content": content
            })
            
        elif role == 'assistant':
            # Check for think block
            think_content = extract_think_block(content)
            
            # Check for tool calls
            tool_calls = extract_tool_calls_from_assistant(content)
            
            # Get remaining text after removing think and tool_call
            remaining_text = remove_think_and_tool_call_tags(content)
            
            # Add think block as assistant message if exists
            if think_content:
                messages.append({
                    "role": "assistant",
                    "content": f"<think>\n{think_content}\n</think>"
                })
            
            # Add tool calls
            for tc in tool_calls:
                messages.append({
                    "role": "tool_call",
                    "content": json.dumps(tc, ensure_ascii=False)
                })
            
            # Add final response if exists (after tool execution or without tools)
            if remaining_text and not tool_calls:
                # This is a final answer without tool calls
                messages.append({
                    "role": "assistant",
                    "content": remaining_text
                })
            elif remaining_text and tool_calls:
                # There might be text after tool calls - handle as separate message
                # But in SWIFT format, tool response comes between, so we skip this
                pass
                
        elif role == 'tool':
            # Tool response
            messages.append({
                "role": "tool_response",
                "content": content
            })
    
    # Try to get tools from sample's 'tools' field if not found in system prompt
    if not tools_str:
        sample_tools = sample.get('tools')
        if sample_tools:
            if isinstance(sample_tools, str):
                try:
                    # Validate it's valid JSON
                    tools = json.loads(sample_tools)
                    tools_str = json.dumps(tools, ensure_ascii=False)
                except json.JSONDecodeError:
                    pass
            elif isinstance(sample_tools, list):
                tools_str = json.dumps(sample_tools, ensure_ascii=False)
    
    # Validate: must have tools and at least user + assistant/tool_call
    if not tools_str:
        return None
    if len(messages) < 2:
        return None
    
    return {
        "tools": tools_str,
        "messages": messages,
        "scenario": sample.get('scenario_category', 'unknown')
    }

def format_hermes_dataset(
    output_dir: str = "data/processed",
    train_size: int = 10000,
    val_size: int = 1000,
    streaming: bool = True
):
    """
    Download and format Hermes Reasoning Tool Use dataset.
    """
    print("Loading Hermes Reasoning Tool Use dataset from HuggingFace...")
    
    if streaming:
        ds = load_dataset(
            "interstellarninja/hermes_reasoning_tool_use",
            split="train",
            streaming=True
        )
    else:
        ds = load_dataset(
            "interstellarninja/hermes_reasoning_tool_use",
            split="train"
        )
    
    train_data = []
    val_data = []
    stats = {
        "single_turn": 0,
        "multi_turn": 0,
        "multi_step": 0,
        "relevance": 0,
        "failed": 0
    }
    
    total_needed = train_size + val_size
    pbar = tqdm(total=total_needed, desc="Processing samples")
    
    debug_shown = False
    
    for i, sample in enumerate(ds):
        if len(train_data) >= train_size and len(val_data) >= val_size:
            break
        
        # Debug: show first sample structure
        if not debug_shown:
            print("\n=== DEBUG: First sample structure ===")
            print(f"Keys: {list(sample.keys())}")
            for key in sample.keys():
                val = sample[key]
                if isinstance(val, str):
                    print(f"{key}: (str) {val[:200]}..." if len(val) > 200 else f"{key}: (str) {val}")
                elif isinstance(val, list):
                    print(f"{key}: (list) {len(val)} items")
                    if val:
                        print(f"  First item: {val[0]}")
                else:
                    print(f"{key}: ({type(val).__name__}) {val}")
            print("=" * 50 + "\n")
            debug_shown = True
            
        converted = convert_hermes_to_swift(sample)
        
        if converted:
            scenario = converted.get('scenario', 'unknown')
            if scenario in stats:
                stats[scenario] += 1
            
            # Split: first train_size go to train, rest to val
            if len(train_data) < train_size:
                train_data.append(converted)
            elif len(val_data) < val_size:
                val_data.append(converted)
                
            pbar.update(1)
        else:
            stats["failed"] += 1
    
    pbar.close()
    
    # Save datasets
    os.makedirs(output_dir, exist_ok=True)
    
    train_path = os.path.join(output_dir, f"hermes_train_{train_size}.json")
    val_path = os.path.join(output_dir, f"hermes_val_{val_size}.json")
    
    with open(train_path, 'w', encoding='utf-8') as f:
        json.dump(train_data, f, indent=2, ensure_ascii=False)
    print(f"Saved {len(train_data)} training samples to {train_path}")
    
    with open(val_path, 'w', encoding='utf-8') as f:
        json.dump(val_data, f, indent=2, ensure_ascii=False)
    print(f"Saved {len(val_data)} validation samples to {val_path}")
    
    # Print statistics
    print("\n=== Dataset Statistics ===")
    for key, value in stats.items():
        print(f"  {key}: {value}")
    
    return train_path, val_path

def preview_sample(sample: Dict[str, Any], max_content_len: int = 200):
    """Pretty print a sample for debugging."""
    print("=" * 60)
    print(f"Scenario: {sample.get('scenario', 'N/A')}")
    print(f"Tools: {sample['tools'][:max_content_len]}...")
    print("\nMessages:")
    for msg in sample['messages']:
        content = msg['content']
        if len(content) > max_content_len:
            content = content[:max_content_len] + "..."
        print(f"  [{msg['role']}]: {content}")
    print("=" * 60)

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Format Hermes dataset for TRM")
    parser.add_argument("--train_size", type=int, default=10000, help="Number of training samples")
    parser.add_argument("--val_size", type=int, default=1000, help="Number of validation samples")
    parser.add_argument("--output_dir", type=str, default="data/processed", help="Output directory")
    parser.add_argument("--preview", action="store_true", help="Preview samples only")
    
    args = parser.parse_args()
    
    if args.preview:
        # Just preview a few samples
        ds = load_dataset(
            "interstellarninja/hermes_reasoning_tool_use",
            split="train",
            streaming=True
        )
        
        for i, sample in enumerate(ds):
            if i >= 5:
                break
            converted = convert_hermes_to_swift(sample)
            if converted:
                preview_sample(converted)
    else:
        format_hermes_dataset(
            output_dir=args.output_dir,
            train_size=args.train_size,
            val_size=args.val_size
        )
