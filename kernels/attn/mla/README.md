# MLA (Multi-head Latent Attention)

Test setup for an MLA kernel, started Feb 2025. **No kernel has been written yet.**

- `gentests.py`: PyTorch reference; `python gentests.py <N> <D> <H_QO> <H_KV> [causal=0|1]` writes a `randn_*_mla.txt` test file
- `harness.impl`: harness to load a test file, check a kernel's output and time it (expects the kernel to define `NUM_WORKERS`)
- `old_gentests.py`: earlier version of `gentests.py`
- `randn_*_mla.txt`: generated test files
