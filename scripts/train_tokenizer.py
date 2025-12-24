"""
Train Tokenizer Separately
"""
import sys
from pathlib import Path
import argparse

sys.path.append(str(Path(__file__).parent.parent / 'src'))

from tokenizer import ToolCallTokenizer


def main():
    parser = argparse.ArgumentParser(description='Train SentencePiece Tokenizer')
    parser.add_argument('--data_path', type=str, default='data/xlam_1k_swift.json',
                        help='Path to training data')
    parser.add_argument('--output_dir', type=str, default='checkpoints',
                        help='Output directory')
    parser.add_argument('--vocab_size', type=int, default=8000,
                        help='Vocabulary size')
    parser.add_argument('--model_type', type=str, default='bpe', choices=['bpe', 'unigram'],
                        help='Tokenizer type')
    
    args = parser.parse_args()
    
    print("=" * 80)
    print("Training SentencePiece Tokenizer")
    print("=" * 80)
    print(f"Data: {args.data_path}")
    print(f"Output: {args.output_dir}")
    print(f"Vocab size: {args.vocab_size}")
    print(f"Model type: {args.model_type}")
    print()
    
    # Train tokenizer
    tokenizer = ToolCallTokenizer()
    tokenizer.train(
        args.data_path,
        args.output_dir,
        vocab_size=args.vocab_size,
        model_type=args.model_type,
    )
    
    print()
    print("=" * 80)
    print("✅ Tokenizer training completed!")
    print("=" * 80)
    print(f"Model saved to: {args.output_dir}/sp_tokenizer.model")
    print(f"Vocabulary size: {tokenizer.vocab_size()}")
    print()
    print("Now you can train the model with:")
    print(f"  python scripts/train.py --tokenizer_path {args.output_dir}/sp_tokenizer.model")


if __name__ == '__main__':
    main()
