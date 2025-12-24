"""
SentencePiece Tokenizer for Tool Calling
"""
import json
import tempfile
import sentencepiece as spm
from pathlib import Path


class ToolCallTokenizer:
    """SentencePiece tokenizer for tool calling task"""
    
    def __init__(self, model_path=None):
        """
        Initialize tokenizer
        
        Args:
            model_path: Path to trained SentencePiece model (.model file)
        """
        self.sp = None
        if model_path and Path(model_path).exists():
            self.load(model_path)
    
    def train(self, data_path, output_dir, vocab_size=8000, model_type='bpe'):
        """
        Train SentencePiece tokenizer from JSONL data
        
        Args:
            data_path: Path to training data (JSON file)
            output_dir: Directory to save tokenizer model
            vocab_size: Vocabulary size
            model_type: 'bpe' or 'unigram'
        """
        import sys
        
        print(f"Training {model_type.upper()} tokenizer with vocab_size={vocab_size}...")
        sys.stdout.flush()
        
        # Extract all text from data
        print("Extracting text from data...", end=' ')
        sys.stdout.flush()
        texts = self._extract_texts(data_path)
        print(f"✓ Extracted {len(texts)} text samples")
        sys.stdout.flush()
        
        # Write to temporary file
        print("Writing to temporary file...", end=' ')
        sys.stdout.flush()
        with tempfile.NamedTemporaryFile(mode='w', delete=False, suffix='.txt', encoding='utf-8') as f:
            for text in texts:
                f.write(text + '\n')
            temp_file = f.name
        print("✓")
        sys.stdout.flush()
        
        # Create output directory
        Path(output_dir).mkdir(parents=True, exist_ok=True)
        
        # Special tokens
        special_tokens = [
            '<PAD>', '<UNK>', '<BOS>', '<EOS>',
            '<TOOL>', '</TOOL>',
        ]
        
        # Train SentencePiece
        print(f"Training SentencePiece model (this may take 1-2 minutes)...", end=' ')
        sys.stdout.flush()
        
        model_prefix = f"{output_dir}/sp_tokenizer"
        spm.SentencePieceTrainer.train(
            input=temp_file,
            model_prefix=model_prefix,
            vocab_size=vocab_size,
            model_type=model_type,
            user_defined_symbols=special_tokens,
            character_coverage=0.9995,
            pad_id=0,
            unk_id=1,
            bos_id=2,
            eos_id=3,
            num_threads=4,
        )
        print("✓")
        sys.stdout.flush()
        
        # Load trained model
        print("Loading trained model...", end=' ')
        sys.stdout.flush()
        self.load(f"{model_prefix}.model")
        print("✓")
        sys.stdout.flush()
        
        print(f"\n✅ Tokenizer ready! Saved to {model_prefix}.model")
        print(f"   Vocabulary size: {self.vocab_size()}")
        sys.stdout.flush()
        
        # Clean up temp file
        Path(temp_file).unlink()
        
        return self
    
    def load(self, model_path):
        """Load trained SentencePiece model"""
        self.sp = spm.SentencePieceProcessor()
        self.sp.load(model_path)
    
    def _extract_texts(self, data_path):
        """Extract all text from JSONL data"""
        texts = []
        
        with open(data_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        for item in data:
            # User messages
            for msg in item['messages']:
                if msg['role'] in ['user', 'tool_call']:
                    texts.append(msg['content'])
            
            # Tools
            texts.append(item['tools'])
        
        return texts
    
    def encode(self, text, add_bos=False, add_eos=False):
        """
        Encode text to token IDs
        
        Args:
            text: Input text
            add_bos: Add BOS token
            add_eos: Add EOS token
        
        Returns:
            List of token IDs
        """
        if self.sp is None:
            raise ValueError("Tokenizer not loaded. Call train() or load() first.")
        
        ids = self.sp.encode(text, out_type=int)
        
        if add_bos:
            ids = [self.bos_id()] + ids
        if add_eos:
            ids = ids + [self.eos_id()]
        
        return ids
    
    def decode(self, ids):
        """Decode token IDs to text"""
        if self.sp is None:
            raise ValueError("Tokenizer not loaded.")
        return self.sp.decode(ids)
    
    def vocab_size(self):
        """Get vocabulary size"""
        if self.sp is None:
            return 0
        return self.sp.get_piece_size()
    
    def pad_id(self):
        """Get PAD token ID"""
        return 0
    
    def unk_id(self):
        """Get UNK token ID"""
        return 1
    
    def bos_id(self):
        """Get BOS token ID"""
        return 2
    
    def eos_id(self):
        """Get EOS token ID"""
        return 3
