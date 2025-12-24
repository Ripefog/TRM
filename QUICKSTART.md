# 🚀 Quick Start - Train TRM Model

## Cách đơn giản nhất (Recommended)

```bash
# Train model Tiny với 1 epoch (test nhanh)
python scripts/quick_train.py --num_epochs 1 --batch_size 4
```

**Thời gian**: 2-3 phút  
**Kết quả**: Model được save vào `checkpoints/quick_model.pt`

---

## Train model Small/Base (Production)

### Bước 1: Chạy lệnh này (chỉ 1 lần)

```bash
python scripts/quick_train.py --num_epochs 1 --batch_size 4
```

Lệnh này sẽ:
- ✅ Train tokenizer (instant)
- ✅ Train model tiny
- ✅ Verify mọi thứ hoạt động

### Bước 2: Sau khi test OK, train model lớn hơn

**Small model** (~32M params):
```bash
python scripts/train_with_config.py --model_size small --num_epochs 50 --batch_size 8
```

**Base model** (~99M params):
```bash
python scripts/train_with_config.py --model_size base --num_epochs 50 --batch_size 4
```

**Lưu ý**: Lệnh này sẽ train tokenizer tự động (đợi 1-2 phút ở bước "Training SentencePiece model...")

---

## Các file quan trọng

```
TRM/
├── data/
│   ├── xlam_1k_swift.json          # Training data
│   └── xlam_val_200_swift.json     # Validation data
├── src/                             # Source code
├── scripts/
│   ├── quick_train.py              # ← Dùng cái này để test
│   └── train_with_config.py        # ← Dùng cái này để train thật
├── checkpoints/                     # Models được save ở đây
└── README.md
```

---

## Troubleshooting

### Nếu training bị treo ở "Training SentencePiece model..."
→ **Đợi 1-2 phút**, nó đang train trong background

### Nếu CUDA out of memory
→ Giảm batch_size:
```bash
python scripts/train_with_config.py --model_size small --batch_size 2
```

### Nếu muốn train nhanh
→ Dùng model tiny:
```bash
python scripts/quick_train.py --num_epochs 1
```

---

## Next Steps

Sau khi train xong, dùng model để inference:

```bash
python scripts/inference.py \
    --model_path checkpoints/small/best_model.pt \
    --tokenizer_path checkpoints/small/sp_tokenizer.model \
    --interactive
```

Good luck! 🚀
