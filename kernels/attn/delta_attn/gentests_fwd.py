#!/usr/bin/env python3
import sys
from tqdm import trange
import torch

def pytorch_ref(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, scale=False):
    if scale:
        q = q * (q.shape[-1] ** -0.5)
    attn = q @ k.transpose(-2, -1)
    # Apply lower-triangular mask
    mask = torch.tril(torch.ones(q.shape[-2], q.shape[-2], dtype=torch.bool, device=q.device))
    attn = attn.masked_fill(~mask, 0)
    o = attn @ v
    return o

# Test dimensions
B = 1
H = 1
N = 1024
D = 64
scale = False

TESTNAME = sys.argv[1] if len(sys.argv) > 1 else 'randn_all'

if TESTNAME.startswith('ones'):
    q = (torch.ones((B, H, N, D), dtype=torch.bfloat16, device='cuda') / D).to(torch.float32)
    k = (torch.ones((B, H, N, D), dtype=torch.bfloat16, device='cuda') / D).to(torch.float32)
    v = (torch.ones((B, H, N, D), dtype=torch.bfloat16, device='cuda') / D).to(torch.float32)
elif TESTNAME.startswith('randn'):
    torch.manual_seed(42)
    q = (torch.randn((B, H, N, D), dtype=torch.bfloat16, device='cuda') / (D**0.5)).to(torch.float32)
    k = (torch.randn((B, H, N, D), dtype=torch.bfloat16, device='cuda') / (D**0.5)).to(torch.float32)
    v = (torch.randn((B, H, N, D), dtype=torch.bfloat16, device='cuda') / D).to(torch.float32)
else:
    print("Invalid test name")
    sys.exit(1)

o = pytorch_ref(q, k, v, scale=scale)

# Flatten each tensor into a list of floats.
q_flat = q.flatten().cpu().numpy().tolist()
k_flat = k.flatten().cpu().numpy().tolist()
v_flat = v.flatten().cpu().numpy().tolist()
o_flat = o.flatten().cpu().numpy().tolist()

filename = f"fwd_{B}x{H}x{N}x{D}.txt"
with open(filename, "w") as f:
    for val in trange(len(q_flat), desc="Writing Q"):
        f.write(f"{q_flat[val]} ")
    for val in trange(len(k_flat), desc="Writing K"):
        f.write(f"{k_flat[val]} ")
    for val in trange(len(v_flat), desc="Writing V"):
        f.write(f"{v_flat[val]} ")
    for val in trange(len(o_flat), desc="Writing O_ref"):
        f.write(f"{o_flat[val]} ")
print(f"Written reference data to {filename}")
