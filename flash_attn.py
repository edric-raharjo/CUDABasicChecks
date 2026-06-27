cat << 'EOF' > test_flash.py
import torch

device = "cuda"
q = torch.randn(1, 8, 1024, 64, dtype=torch.float16, device=device)
k = torch.randn(1, 8, 1024, 64, dtype=torch.float16, device=device)
v = torch.randn(1, 8, 1024, 64, dtype=torch.float16, device=device)

print(f"Using PyTorch version: {torch.__version__}")

with torch.nn.attention.sdp_kernel(enable_flash=True, enable_math=False, enable_mem_efficient=False):
    try:
        out = torch.nn.functional.scaled_dot_product_attention(q, k, v)
        print("⚡ Success! FlashAttention is working perfectly inside your venv.")
    except Exception as e:
        print(f"❌ FlashAttention failed: {e}")
EOF

python3 test_flash.py
