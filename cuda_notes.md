# Engineering Notes: FlashAttention Installation Debrief (Ubuntu)
**Date:** June 28, 2026
**Environment:** Ubuntu Linux | Python 3.14 | PyTorch (Virtual Env)

---

### Error 1: Missing Host CUDA Compiler & Paths
* **Symptom:** `flash_attn was requested, but nvcc was not found.`
  `OSError: CUDA_HOME environment variable is not set.`
* **Root Cause:** PyTorch wheels ship with their own runtime CUDA binaries so models can run out of the box. However, they do *not* include development headers or the actual CUDA compiler (`nvcc`). Because a matching pre-built wheel wasn't available for Python 3.14 on PyPI, `pip` tried to compile FlashAttention from source but failed because the local system lacked the host compilation environment.
* **Resolution:** 1. Installed the system-level toolkit: `sudo apt install nvidia-cuda-toolkit`
  2. Exposed the paths to the environment:
     ```bash
     export CUDA_HOME=/usr/lib/nvidia-cuda-toolkit
     export PATH=$CUDA_HOME/bin:$PATH
     ```

---

### Error 2: Host C++ Compiler Version Mismatch
* **Symptom:**
  `RuntimeError: The current installed version of x86_64-linux-gnu-g++ (15.2.0) is greater than the maximum required version by CUDA 12.4. Please make sure to use an adequate version of x86_64-linux-gnu-g++ (>=6.0.0, <14.0).`
* **Root Cause:**
  NVIDIA's CUDA compilation framework sets strict upper limits on the host's GNU C++ compiler (`g++`) version to guarantee compatibility with its internal syntax parsing. The host system was running cutting-edge `g++ 15.2.0`, but the CUDA toolkit required an older version (`< 14.0`).
* **Resolution:**
  Forced the build pipeline to use an explicitly compatible, older version of the compiler already present on the system (`v13`) via environment overrides:
  ```bash
  export CC=/usr/bin/gcc-13
  export CXX=/usr/bin/g++-13
