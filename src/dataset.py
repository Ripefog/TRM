"""
Dataset for Tool Calling Task
"""
import json
import torch
from torch.utils.data import Dataset


class ToolCallDataset(Dataset):
    """Dataset for tool calling task"""
    
    def __init__(self, data_path, tokenizer, max_len=512, max_args_len=128, tool_to_id=None):
        """
        Initialize dataset
        
        Args:
            data_path: Path to JSON data file
            tokenizer: ToolCallTokenizer instance
            max_len: Maximum input sequence length
            max_args_len: Maximum arguments sequence length
            tool_to_id: Optional pre-built tool vocabulary dict. If None, will build from data.
        """
        self.tokenizer = tokenizer
        self.max_len = max_len
        self.max_args_len = max_args_len
        
        # Load data
        print(f"Loading data from {data_path}...")
        with open(data_path, 'r', encoding='utf-8') as f:
            self.data = json.load(f)
        
        # Build or use provided tool vocabulary
        if tool_to_id is not None:
            print(f"Using provided tool vocabulary ({len(tool_to_id)} tools)")
            self.tool_to_id = tool_to_id
            self.id_to_tool = {v: k for k, v in tool_to_id.items()}
        else:
            print("Building tool vocabulary from data...")
            self.tool_to_id = {}
            self.id_to_tool = {}
            self._build_tool_vocab()
        
        print(f"Dataset loaded:")
        print(f"  - Samples: {len(self.data)}")
        print(f"  - Unique tools: {len(self.tool_to_id)}")
        print(f"  - Max input length: {max_len}")
        print(f"  - Max args length: {max_args_len}")
    
    def _build_tool_vocab(self):
        """Build tool name → tool_id mapping"""
        all_tools = set()
        
        for item in self.data:
            tools_json = json.loads(item['tools'])
            for tool in tools_json:
                # Handle OpenAI format
                if 'type' in tool and tool['type'] == 'function':
                    tool_name = tool['function']['name']
                else:
                    tool_name = tool.get('name', 'unknown')
                all_tools.add(tool_name)
        
        # Create mapping (sorted for consistency)
        for idx, tool_name in enumerate(sorted(all_tools)):
            self.tool_to_id[tool_name] = idx
            self.id_to_tool[idx] = tool_name
        
        print(f"\nTool vocabulary ({len(all_tools)} tools):")
        for tool_name, tool_id in sorted(self.tool_to_id.items(), key=lambda x: x[1])[:10]:
            print(f"  {tool_id}: {tool_name}")
        if len(all_tools) > 10:
            print(f"  ... and {len(all_tools) - 10} more")
    
    def _format_input(self, user_content, tools_json):
        """
        Format input text
        
        Format:
            Tools: [tool1, tool2, ...]
            User: {user query}
        """
        # Parse tools
        tools = json.loads(tools_json)
        tool_names = []
        for tool in tools:
            if 'type' in tool and tool['type'] == 'function':
                tool_names.append(tool['function']['name'])
            else:
                tool_names.append(tool.get('name', 'unknown'))
        
        # Format
        input_text = f"Tools: {', '.join(tool_names)}\n"
        input_text += f"User: {user_content}"
        
        return input_text
    
    def _parse_tool_call(self, tool_call_content):
        """
        Parse tool call content
        
        Input: '{"name": "web_chain_details", "arguments": {"chain_slug": "ethereum"}}'
        
        Returns:
            tool_name: str
            arguments: dict
        """
        try:
            tool_call = json.loads(tool_call_content)
            tool_name = tool_call.get('name', 'unknown')
            arguments = tool_call.get('arguments', {})
            return tool_name, arguments
        except json.JSONDecodeError:
            print(f"Warning: Failed to parse tool call: {tool_call_content}")
            return 'unknown', {}
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        """
        Get a single training sample
        
        Returns:
            {
                'input_ids': List[int],
                'target_tool_id': int,
                'target_args_ids': List[int],
            }
        """
        item = self.data[idx]
        
        # Extract messages
        user_msg = None
        tool_call_msg = None
        for msg in item['messages']:
            if msg['role'] == 'user':
                user_msg = msg['content']
            elif msg['role'] == 'tool_call':
                tool_call_msg = msg['content']
        
        if user_msg is None or tool_call_msg is None:
            raise ValueError(f"Sample {idx} missing user or tool_call message")
        
        # Format input
        input_text = self._format_input(user_msg, item['tools'])
        
        # Parse tool call
        tool_name, arguments = self._parse_tool_call(tool_call_msg)
        tool_id = self.tool_to_id.get(tool_name, 0)  # Default to 0 if unknown
        
        # Tokenize input
        input_ids = self.tokenizer.encode(input_text, add_bos=True, add_eos=False)
        input_ids = input_ids[:self.max_len]  # Truncate
        
        # Tokenize arguments (as JSON string)
        args_text = json.dumps(arguments)
        target_args_ids = self.tokenizer.encode(args_text, add_bos=False, add_eos=True)
        target_args_ids = target_args_ids[:self.max_args_len]  # Truncate
        
        return {
            'input_ids': input_ids,
            'target_tool_id': tool_id,
            'target_args_ids': target_args_ids,
        }
    
    def get_stats(self):
        """Get dataset statistics"""
        tool_counts = {}
        input_lengths = []
        args_lengths = []
        
        for item in self.data:
            # Tool counts
            for msg in item['messages']:
                if msg['role'] == 'tool_call':
                    tool_name, _ = self._parse_tool_call(msg['content'])
                    tool_counts[tool_name] = tool_counts.get(tool_name, 0) + 1
            
            # Lengths
            sample = self[len(input_lengths)]
            input_lengths.append(len(sample['input_ids']))
            args_lengths.append(len(sample['target_args_ids']))
        
        print("\n=== Dataset Statistics ===")
        print(f"Total samples: {len(self.data)}")
        print(f"\nInput lengths:")
        print(f"  Mean: {sum(input_lengths) / len(input_lengths):.1f}")
        print(f"  Max: {max(input_lengths)}")
        print(f"  Min: {min(input_lengths)}")
        print(f"\nArguments lengths:")
        print(f"  Mean: {sum(args_lengths) / len(args_lengths):.1f}")
        print(f"  Max: {max(args_lengths)}")
        print(f"  Min: {min(args_lengths)}")
        print(f"\nTop 10 tools:")
        for tool_name, count in sorted(tool_counts.items(), key=lambda x: -x[1])[:10]:
            print(f"  {tool_name}: {count}")
        
        return {
            'tool_counts': tool_counts,
            'input_lengths': input_lengths,
            'args_lengths': args_lengths,
        }
