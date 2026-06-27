import torch

print("🔮 Checking your GPU setup...")
print(f"🐍 PyTorch Version: {torch.__version__}")

if torch.cuda.is_available():
    print("🎉 SUCCESS! CUDA is available! 🎉")
    print(f"🚀 Device Name: {torch.cuda.get_device_name(0)}")
    print(f"🔥 Total VRAM: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.2f} GB")
    
    # Quick sanity test tensor
    x = torch.tensor([1.0, 2.0, 3.0]).cuda()
    print(f"💪 Tested a tensor on device: {x.device}")
else:
    print("❌ Bummer... CUDA is NOT available. PyTorch can't see your GPU. 😢")
