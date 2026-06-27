import flash_attn
import torch

print("🌟✨ FLASH ATTENTION DIAGNOSTIC CENTER ✨🌟")
print("=" * 60)

print(f"⚡ FlashAttention Version: {flash_attn.__version__}")
print(f"🐍 PyTorch Version: {torch.__version__}")

print("\n🔍 Checking CUDA availability...")

if torch.cuda.is_available():
    print("🎉🎉🎉 SUCCESS! CUDA IS AVAILABLE! 🎉🎉🎉")

    gpu_name = torch.cuda.get_device_name(0)
    gpu_props = torch.cuda.get_device_properties(0)

    print(f"🚀 GPU Name: {gpu_name}")
    print(f"🔥 Total VRAM: {gpu_props.total_memory / 1024**3:.2f} GB")
    print(f"🧠 CUDA Capability: {gpu_props.major}.{gpu_props.minor}")
    print(f"⚙️ Multiprocessors: {gpu_props.multi_processor_count}")
    print(f"📍 Current Device Index: {torch.cuda.current_device()}")

    print("\n🧪 Running GPU sanity test...")
    x = torch.randn(1024, 1024, device="cuda")
    y = torch.randn(1024, 1024, device="cuda")
    z = x @ y

    print(f"💪 Tensor Device: {z.device}")
    print(f"📊 Tensor Shape: {tuple(z.shape)}")
    print("✅ Matrix multiplication successful!")
    print("🚀 GPU computation confirmed!")

    print("\n⚡ FlashAttention appears importable and ready!")
    print("🎯 Your AI workloads are cleared for takeoff! 🚀🚀🚀")

else:
    print("❌❌❌ CUDA IS NOT AVAILABLE ❌❌❌")
    print("😢 PyTorch cannot detect an NVIDIA GPU.")
    print("🔧 Check:")
    print("   • NVIDIA driver installation 🪛")
    print("   • CUDA toolkit compatibility ⚙️")
    print("   • PyTorch CUDA build 🐍")
    print("   • GPU visibility via nvidia-smi 📈")

print("\n🏁 Diagnostic complete!")
