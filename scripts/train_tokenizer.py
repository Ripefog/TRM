"""
Train BPE tokenizer on Hermes Tool Calling data.

The tokenizer needs to handle:
- ChatML format (<|im_start|>, <|im_end|>)
- Tool calling tags (<tool_call>, </tool_call>, <tools>, </tools>)
- Thinking tags (<think>, </think>)
- JSON content in tool calls
"""

import json
import os
import argparse
from tokenizers import Tokenizer, models, trainers, pre_tokenizers, processors, decoders
from typing import List, Iterator

def extract_texts_from_hermes(data_path: str) -> Iterator[str]:
    """
    Extract all text content from Hermes format data for tokenizer training.
    """
    with open(data_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    for item in data:
        # Yield tools JSON
        yield item['tools']
        
        # Yield message contents
        for msg in item['messages']:
            yield msg['content']

def train_tokenizer(
    data_path: str,
    output_path: str,
    vocab_size: int = 32000,
    min_frequency: int = 2,
):
    """
    Train a BPE tokenizer with special tokens for tool calling.
    """
    print(f"Training tokenizer on {data_path}")
    print(f"Vocab size: {vocab_size}")
    
    # Initialize tokenizer with BPE
    tokenizer = Tokenizer(models.BPE(unk_token="<unk>"))
    
    # Pre-tokenizer: split on whitespace and punctuation
    tokenizer.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
    
    # Define special tokens
    special_tokens = [
        "<pad>",      # 0: Padding
        "<s>",        # 1: BOS
        "</s>",       # 2: EOS
        "<unk>",      # 3: Unknown
        # ChatML tokens
        "<|im_start|>",
        "<|im_end|>",
        # Tool calling tokens
        "<tools>",
        "</tools>",
        "<tool_call>",
        "</tool_call>",
        "<tool_response>",
        "</tool_response>",
        # Thinking tokens
        "<think>",
        "</think>",
        # Role tokens
        "system",
        "user",
        "assistant",
    ]
    
    # Trainer
    trainer = trainers.BpeTrainer(
        vocab_size=vocab_size,
        min_frequency=min_frequency,
        special_tokens=special_tokens,
        show_progress=True,
    )
    
    # Extract texts
    texts = list(extract_texts_from_hermes(data_path))
    print(f"Training on {len(texts)} text samples")
    
    # Train
    tokenizer.train_from_iterator(texts, trainer)
    
    # Post-processor: add BOS/EOS
    tokenizer.post_processor = processors.ByteLevel(trim_offsets=False)
    
    # Decoder
    tokenizer.decoder = decoders.ByteLevel()
    
    # Save
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    tokenizer.save(output_path)
    print(f"Tokenizer saved to {output_path}")
    
    # Test
    print("\n=== Tokenizer Test ===")
    test_texts = [
        "<|im_start|>user\nWhat's the weather?<|im_end|>",
        '<tool_call>\n{"name": "get_weather", "arguments": {"city": "Tokyo"}}\n</tool_call>',
        "<think>\nI need to use the weather API.\n</think>",
    ]
    
    for text in test_texts:
        encoded = tokenizer.encode(text)
        decoded = tokenizer.decode(encoded.ids)
        print(f"Original: {text[:50]}...")
        print(f"Tokens: {len(encoded.ids)}")
        print(f"Decoded: {decoded[:50]}...")
        print()
    
    # Print vocab info
    print(f"Final vocab size: {tokenizer.get_vocab_size()}")
    
    # Verify special tokens
    print("\nSpecial token IDs:")
    for token in special_tokens:
        token_id = tokenizer.token_to_id(token)
        print(f"  {token}: {token_id}")
    
    return tokenizer

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train tokenizer")
    parser.add_argument("--data_path", type=str, 
                       default="data/processed/hermes_train_10000.json")
    parser.add_argument("--output_path", type=str,
                       default="data/processed/tokenizer.json")
    parser.add_argument("--vocab_size", type=int, default=32000)
    parser.add_argument("--min_frequency", type=int, default=2)
    
    args = parser.parse_args()
    
    train_tokenizer(
        data_path=args.data_path,
        output_path=args.output_path,
        vocab_size=args.vocab_size,
        min_frequency=args.min_frequency,
    )
