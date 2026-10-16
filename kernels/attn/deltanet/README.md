# DeltaNet (delta-rule linear attention) in ThunderKittens

Chunked forward kernel for DeltaNet, written Feb–Jul 2025.

## Status

| File | What | State |
|---|---|---|
| `4090.cu` | Chunked forward kernel, RTX 4090 | Correct on all-ones input with masking disabled. The `T` mask code exists (May 2025) but hasn't been checked on random input. |
| `h100.cu` | Port of `4090.cu` with more workers and tiles for H100 | Never run ("zero shot?" commit) |
| (none) | Backward kernel | Written Feb–Mar 2025, now commented out in `archive/delta_attn_og.cu`; not ported to the chunked kernel |

## Layout

```
4090.cu, h100.cu     kernels (one per GPU)
fwd_harness.impl     loads a test file, runs the kernel, compares to reference, times it
gentests_fwd.py      PyTorch reference (chunk_delta_rule_forward) -> test .txt file
Makefile             GPU_TARGET=4090 (default) or H100
reference/           code we built on / compared against:
                       alxndrTL_lin_attn_4090.cu  linear-attention kernel (fwd+bwd harnesses), our starting point
                       alxndrTL_code.cu           same kernel with the harness includes commented out
                       triton_deltanet.py         Triton DeltaNet reference
archive/             earlier WIP kernels, test generators and harnesses (kept for reference, not built)
```

## Running

```bash
python gentests_fwd.py            # writes debug_delta_fwd_{B}x{H}x{N}x{D}.txt (B, H, N, D set at top of file)
make                              # or: make GPU_TARGET=H100
./deltanet debug_delta_fwd_8x8x128x16.txt
```

The harness writes `printouts/o.txt`, `printouts/o_ref.txt` and `printouts/diff.txt`.

The kernel's `ATTN_D`, `NUM_WORKERS` and `ACTIVE_TILES` constants (top of each `.cu`) must match the
`D` used in `gentests_fwd.py`.
