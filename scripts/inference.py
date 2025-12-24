"""
Inference Script for TRM Tool Calling
"""
import torch
import json
import argparse
from pathlib import Path
import sys

sys.path.append(str(Path(__file__).parent.parent))

from src.tokenizer import ToolCallTokenizer
from src.model import SimpleTRMToolCalling


def load_model(model_path, tokenizer_path, device='cuda'):
    """Load trained model and tokenizer"""
    print(f"Loading model from {model_path}...")
    
    # Load checkpoint
    checkpoint = torch.load(model_path, map_location=device)
    
    # Load tokenizer
    tokenizer = ToolCallTokenizer(tokenizer_path)
    
    # Create model
    model_config = checkpoint['model_config']
    tool_to_id = checkpoint['tool_to_id']
    id_to_tool = {v: k for k, v in tool_to_id.items()}
    
    model = SimpleTRMToolCalling(
        vocab_size=tokenizer.vocab_size(),
        num_tools=len(tool_to_id),
        **model_config
    ).to(device)
    
    # Load weights
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    
    print(f"✓ Model loaded ({sum(p.numel() for p in model.parameters())/1e6:.1f}M params)")
    print(f"✓ Tokenizer loaded (vocab_size={tokenizer.vocab_size()})")
    print(f"✓ Tool vocabulary: {len(tool_to_id)} tools")
    
    return model, tokenizer, tool_to_id, id_to_tool


def format_input(tools_json, user_query):
    """Format input for model"""
    # Parse tools
    tools = json.loads(tools_json) if isinstance(tools_json, str) else tools_json
    tool_names = []
    for tool in tools:
        if 'type' in tool and tool['type'] == 'function':
            tool_names.append(tool['function']['name'])
        else:
            tool_names.append(tool.get('name', 'unknown'))
    
    # Format
    input_text = f"Tools: {', '.join(tool_names)}\n"
    input_text += f"User: {user_query}"
    
    return input_text


def predict(model, tokenizer, tool_to_id, id_to_tool, tools_json, user_query, device='cuda'):
    """Predict tool call for a query"""
    # Format input
    input_text = format_input(tools_json, user_query)
    
    # Tokenize
    input_ids = tokenizer.encode(input_text, add_bos=True, add_eos=False)
    input_ids = torch.tensor([input_ids], dtype=torch.long).to(device)
    
    # Forward pass
    with torch.no_grad():
        outputs_per_step = model(input_ids, attention_mask=None, target_args_ids=None)
    
    # Get final predictions
    final_outputs = outputs_per_step[-1]
    
    # Tool selection
    tool_logits = final_outputs['tool_logits']
    tool_probs = torch.softmax(tool_logits, dim=-1)
    tool_id = tool_logits.argmax(dim=-1).item()
    tool_name = id_to_tool[tool_id]
    tool_confidence = tool_probs[0, tool_id].item()
    
    # Correctness prediction
    q_logit = final_outputs['q_logit']
    q_prob = torch.sigmoid(q_logit).item()
    
    # Generate arguments (greedy decoding)
    # Start with BOS token
    args_ids = [tokenizer.bos_id()]
    max_len = 128
    
    for _ in range(max_len):
        # Prepare input
        args_input = torch.tensor([args_ids], dtype=torch.long).to(device)
        
        # Forward (with current args as input)
        with torch.no_grad():
            outputs = model(input_ids, attention_mask=None, target_args_ids=args_input)
        
        # Get logits for next token
        if 'args_logits' in outputs[-1]:
            args_logits = outputs[-1]['args_logits']
            next_token_logits = args_logits[0, -1, :]
            next_token_id = next_token_logits.argmax().item()
            
            # Stop if EOS
            if next_token_id == tokenizer.eos_id():
                break
            
            args_ids.append(next_token_id)
        else:
            break
    
    # Decode arguments
    args_text = tokenizer.decode(args_ids)
    
    # Try to parse as JSON
    try:
        arguments = json.loads(args_text)
    except json.JSONDecodeError:
        arguments = {"raw": args_text}
    
    return {
        'tool': tool_name,
        'tool_confidence': tool_confidence,
        'arguments': arguments,
        'correctness_score': q_prob,
        'num_steps': len(outputs_per_step),
    }


def main():
    parser = argparse.ArgumentParser(description='TRM Tool Calling Inference')
    parser.add_argument('--model_path', type=str, required=True, help='Path to model checkpoint')
    parser.add_argument('--tokenizer_path', type=str, required=True, help='Path to tokenizer model')
    parser.add_argument('--tools', type=str, help='Tools JSON string or path to JSON file')
    parser.add_argument('--query', type=str, help='User query')
    parser.add_argument('--interactive', action='store_true', help='Interactive mode')
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu')
    
    args = parser.parse_args()
    
    # Load model
    model, tokenizer, tool_to_id, id_to_tool = load_model(args.model_path, args.tokenizer_path, args.device)
    
    print("\n" + "=" * 80)
    print("TRM Tool Calling - Inference")
    print("=" * 80)
    
    if args.interactive:
        # Interactive mode
        print("\nInteractive mode. Type 'quit' to exit.")
        print("Available tools:", list(tool_to_id.keys())[:10], "..." if len(tool_to_id) > 10 else "")
        print()
        
        while True:
            # Get tools
            tools_input = input("\nTools (JSON or comma-separated names): ").strip()
            if tools_input.lower() == 'quit':
                break
            
            # Parse tools
            if tools_input.startswith('[') or tools_input.startswith('{'):
                tools_json = tools_input
            else:
                # Simple format: "tool1, tool2, tool3"
                tool_names = [t.strip() for t in tools_input.split(',')]
                tools_json = json.dumps([{"name": name} for name in tool_names])
            
            # Get query
            query = input("Query: ").strip()
            if query.lower() == 'quit':
                break
            
            # Predict
            result = predict(model, tokenizer, tool_to_id, id_to_tool, tools_json, query, args.device)
            
            # Print result
            print("\n" + "-" * 80)
            print(f"Tool: {result['tool']} (confidence: {result['tool_confidence']:.3f})")
            print(f"Arguments: {json.dumps(result['arguments'], indent=2)}")
            print(f"Correctness score: {result['correctness_score']:.3f}")
            print(f"Reasoning steps: {result['num_steps']}")
            print("-" * 80)
    
    else:
        # Single prediction mode
        if not args.tools or not args.query:
            print("Error: --tools and --query required in non-interactive mode")
            return
        
        # Load tools
        if Path(args.tools).exists():
            with open(args.tools, 'r') as f:
                tools_json = f.read()
        else:
            tools_json = args.tools
        
        # Predict
        result = predict(model, tokenizer, tool_to_id, id_to_tool, tools_json, args.query, args.device)
        
        # Print result
        print("\nPrediction:")
        print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
