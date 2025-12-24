"""
Test New Model Structure
"""
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).parent.parent))

from src.models import create_model, print_model_info

print("=" * 80)
print("Testing New Model Structure")
print("=" * 80)

# Test 1: Create models of different sizes
print("\n1. Creating models...")

for size in ['tiny', 'small', 'base']:
    print(f"\n{size.upper()} model:")
    model = create_model(size, vocab_size=8000, num_tools=150)
    print_model_info(model)

# Test 2: Create model with custom config
print("\n2. Creating custom model...")
model = create_model(
    'small',
    vocab_size=8000,
    num_tools=150,
    use_swiglu=False,  # Override
    deep_recursion_steps=5  # Override
)
print_model_info(model)

# Test 3: Forward pass
print("\n3. Testing forward pass...")
import torch

batch_size = 2
seq_len = 64
args_len = 32

input_ids = torch.randint(0, 8000, (batch_size, seq_len))
target_args_ids = torch.randint(0, 8000, (batch_size, args_len))

outputs = model(input_ids, target_args_ids=target_args_ids)

print(f"✓ Forward pass successful")
print(f"  Supervision steps: {len(outputs)}")
print(f"  Tool logits shape: {outputs[0]['tool_logits'].shape}")
print(f"  Args logits shape: {outputs[-1]['args_logits'].shape}")

# Test 4: Backward pass
print("\n4. Testing backward pass...")
loss = 0
for step_outputs in outputs:
    tool_logits = step_outputs['tool_logits']
    target_tool_ids = torch.randint(0, 150, (batch_size,))
    loss += torch.nn.functional.cross_entropy(tool_logits, target_tool_ids)

loss.backward()
print(f"✓ Backward pass successful (loss: {loss.item():.4f})")

print("\n" + "=" * 80)
print("✅ All tests passed!")
print("=" * 80)
print("\nNew model structure working correctly!")
print("You can now use: from src.models import create_model")
