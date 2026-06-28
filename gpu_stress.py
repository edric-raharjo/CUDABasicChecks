import torch
import time
import sys

# Quick sanity check
if not torch.cuda.is_available():
    print("Error: CUDA is not available in this PyTorch installation.")
    sys.exit(1)

device = torch.device("cuda")
print(f"Using GPU Device: {torch.cuda.get_device_name(0)}")
print(f"PyTorch CUDA Version: {torch.version.cuda}")

# Matrix size optimized to consume ~7.2 GB of VRAM total for 3 matrices in float32
# (16000 * 16000 * 4 bytes) ≈ 1.02 GB per tensor. We allocate multiple or perform large ops.
MATRIX_SIZE = 22000 

print(f"\nAllocating heavy matrices ({MATRIX_SIZE}x{MATRIX_SIZE})...")
try:
    # Creating two massive random matrices directly on the GPU
    A = torch.randn(MATRIX_SIZE, MATRIX_SIZE, dtype=torch.float32, device=device)
    B = torch.randn(MATRIX_SIZE, MATRIX_SIZE, dtype=torch.float32, device=device)
    
    allocated = torch.cuda.memory_allocated(0) / (1024 ** 3)
    print(f"Successfully allocated! Current VRAM usage: {allocated:.2f} GB")
except RuntimeError as e:
    print(f"OOM Error: Lower MATRIX_SIZE slightly if your OS is eating too much VRAM: {e}")
    sys.exit(1)

print("\nSlamming the GPU. Press Ctrl+C to stop the stress test.")
print("Check your power draw in another terminal using nvidia-smi!")
print("-" * 60)

count = 0
start_time = time.time()

try:
    while True:
        # Perform matrix multiplication
        # We assign back to a dummy variable to keep the memory occupied
        C = torch.matmul(A, B)
        
        # CRITICAL: CUDA operations are asynchronous in Python. 
        # We must force a sync step to compel the script to wait for the GPU to finish,
        # otherwise the loop spins on the CPU and doesn't stress the hardware rails fully.
        torch.cuda.synchronize()
        
        count += 1
        if count % 10 == 0:
            elapsed = time.time() - start_time
            print(f"Iterations completed: {count} | Elapsed time: {elapsed:.1f}s", end="\r")

except KeyboardInterrupt:
    print("\n\nTest stopped by user.")
    # Free up the allocations cleanly
    del A, B, C
    torch.cuda.empty_cache()
    print("VRAM cleared.")
