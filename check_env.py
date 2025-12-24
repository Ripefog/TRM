import sys
print("Python version:", sys.version)
print("Python executable:", sys.executable)

try:
    import torch
    print("PyTorch version:", torch.__version__)
    print("CUDA available:", torch.cuda.is_available())
except Exception as e:
    print("PyTorch import error:", e)

try:
    import sentencepiece
    print("SentencePiece version:", sentencepiece.__version__)
except Exception as e:
    print("SentencePiece import error:", e)

print("\nTest completed!")
