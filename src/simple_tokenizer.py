"""
Simple Character-Level Tokenizer (No SentencePiece needed)
Fast alternative for testing
"""
import json
from collections import Counter
from pathlib import Path


class SimpleTokenizer:
    """Simple character-level tokenizer for quick testing"""
    
    def __init__(self, vocab_size=2000):
        self.vocab_size = vocab_size
        self.char_to_id = {}
        self.id_to_char = {}
        self.trained = False
    
    def train(self, data_path, output_dir):
        """Train tokenizer by building character vocabulary"""
        print(f"Training simple tokenizer (vocab_size={self.vocab_size})...")
        
        # Extract all text
        print("  Extracting text...", end=' ')
        texts = []
        with open(data_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        for item in data:
            for msg in item['messages']:
                texts.append(msg['content'])
            texts.append(item['tools'])
        print(f"✓ ({len(texts)} samples)")
        
        # Count characters
        print("  Counting characters...", end=' ')
        all_text = ''.join(texts)
        char_counts = Counter(all_text)
        print(f"✓ ({len(char_counts)} unique chars)")
        
        # Build vocabulary (most common characters)
        print("  Building vocabulary...", end=' ')
        # Reserve IDs for special tokens
        self.char_to_id = {
            '<PAD>': 0,
            '<UNK>': 1,
            '<BOS>': 2,
            '<EOS>': 3,
        }
        
        # Add most common characters
        most_common = char_counts.most_common(self.vocab_size - 4)
        for idx, (char, _) in enumerate(most_common, start=4):
            self.char_to_id[char] = idx
        
        # Reverse mapping
        self.id_to_char = {v: k for k, v in self.char_to_id.items()}
        self.trained = True
        print(f"✓ ({len(self.char_to_id)} tokens)")
        
        # Save
        Path(output_dir).mkdir(parents=True, exist_ok=True)
        vocab_path = Path(output_dir) / 'simple_tokenizer_vocab.json'
        with open(vocab_path, 'w', encoding='utf-8') as f:
            json.dump(self.char_to_id, f, ensure_ascii=False, indent=2)
        
        print(f"\n✅ Tokenizer saved to {vocab_path}")
        return self
    
    def load(self, vocab_path):
        """Load vocabulary"""
        with open(vocab_path, 'r', encoding='utf-8') as f:
            self.char_to_id = json.load(f)
        self.id_to_char = {v: k for k, v in self.char_to_id.items()}
        self.trained = True
    
    def encode(self, text, add_bos=False, add_eos=False):
        """Encode text to IDs"""
        if not self.trained:
            raise ValueError("Tokenizer not trained!")
        
        ids = []
        if add_bos:
            ids.append(2)  # BOS
        
        for char in text:
            ids.append(self.char_to_id.get(char, 1))  # UNK if not in vocab
        
        if add_eos:
            ids.append(3)  # EOS
        
        return ids
    
    def decode(self, ids):
        """Decode IDs to text"""
        chars = []
        for id in ids:
            char = self.id_to_char.get(id, '')
            if char not in ['<PAD>', '<UNK>', '<BOS>', '<EOS>']:
                chars.append(char)
        return ''.join(chars)
    
    def vocab_size_actual(self):
        """Get actual vocabulary size"""
        return len(self.char_to_id)
    
    def pad_id(self):
        return 0
    
    def unk_id(self):
        return 1
    
    def bos_id(self):
        return 2
    
    def eos_id(self):
        return 3
