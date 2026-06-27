# PyTorch + CUDA 12.9 + FlashAttention 2.8.3 Setup Summary

## Goal

Set up:

- PyTorch 2.8.0 + CUDA 12.9
- FlashAttention 2.8.3
- NVIDIA GeForce RTX 5050 Laptop GPU
- Python 3.12 virtual environment (`DS312`)

---

## Step 1: Install PyTorch CUDA 12.9

Command:

```bash
pip install torch==2.8.0 torchvision==0.23.0 torchaudio==2.8.0 \
    --index-url https://download.pytorch.org/whl/cu129
```

Result:

- Downloaded CUDA 12.9 runtime packages
- Installed:
  - torch 2.8.0+cu129
  - torchvision 0.23.0+cu129
  - torchaudio 2.8.0+cu129
  - Triton 3.4.0
  - CUDA libraries

Success.

---

## Step 2: Verify CUDA

Command:

```bash
python3 cuda_test/torch_cuda_check.py
```

Output:

```text
PyTorch Version: 2.8.0+cu129
CUDA is available!
Device Name: NVIDIA GeForce RTX 5050 Laptop GPU
VRAM: 7.53 GB
```

Success.

---

## Step 3: First FlashAttention Installation Attempt

Command:

```bash
pip install flash-attn==2.8.3
```

Error:

```text
ModuleNotFoundError: No module named 'torch'
```

Cause:

- FlashAttention's build process ran inside an isolated build environment.
- That temporary environment did not contain PyTorch.

---

## Attempted Fix

Command:

```bash
pip install --no-build-isolation flash-attn==2.8.3
```

Result:

- Torch became visible.
- Build progressed further.

New error:

```text
error: [Errno 18] Invalid cross-device link
```

while processing:

```text
flash_attn-2.8.3+cu12torch2.8cxx11abiTRUE-cp312-cp312-linux_x86_64.whl
```

Cause:

- FlashAttention located a prebuilt wheel.
- Installer failed while moving/renaming the wheel between filesystems.

---

## Solution: Install the Wheel Manually

The build log revealed the exact wheel URL:

```text
https://github.com/Dao-AILab/flash-attention/releases/download/v2.8.3/flash_attn-2.8.3+cu12torch2.8cxx11abiTRUE-cp312-cp312-linux_x86_64.whl
```

Download:

```bash
wget https://github.com/Dao-AILab/flash-attention/releases/download/v2.8.3/flash_attn-2.8.3+cu12torch2.8cxx11abiTRUE-cp312-cp312-linux_x86_64.whl
```

Install:

```bash
pip install flash_attn-2.8.3+cu12torch2.8cxx11abiTRUE-cp312-cp312-linux_x86_64.whl
```

Result:

```text
Successfully installed flash-attn-2.8.3
```

Success.

---

## Step 4: Verify FlashAttention

Command:

```bash
python3 cuda_test/flash_test.py
```

Output:

```text
Flash Attention version: 2.8.3
Device: NVIDIA GeForce RTX 5050 Laptop GPU
```

Success.

---

# Final Working Stack

| Component | Version |
|------------|---------|
| Python | 3.12 |
| PyTorch | 2.8.0+cu129 |
| TorchVision | 0.23.0+cu129 |
| TorchAudio | 2.8.0+cu129 |
| CUDA Runtime | 12.9 |
| Triton | 3.4.0 |
| FlashAttention | 2.8.3 |
| GPU | NVIDIA GeForce RTX 5050 Laptop GPU |

---

# Lessons Learned

1. `flash-attn` initially failed because build isolation hid the installed PyTorch.
2. `--no-build-isolation` fixed the Torch visibility issue.
3. The next failure (`Invalid cross-device link`) was not a CUDA or PyTorch problem.
4. FlashAttention already had a matching prebuilt wheel for:
   - Python 3.12
   - PyTorch 2.8
   - CUDA 12.x
5. Downloading and installing the wheel manually bypassed the filesystem bug and completed installation successfully.
