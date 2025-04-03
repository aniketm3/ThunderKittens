#!/usr/bin/env python3
import sys
from tqdm import trange
import torch

def delta_rule_recurrence(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, beta: torch.Tensor, initial_state=None, output_final_state=True):
    """
    Implements a simple delta attention recurrence.
    
    Given inputs q, k, v (shape (B, H, L, D)) and a beta (shape (B, H, L)),
    we:
      1. Scale q by D^{-0.5}.
      2. Initialize a memory state S (shape (B, H, D, D)), starting at zero (or an initial state).
      3. For each time step i from 0 to L-1:
           - Extract _q, _k, _v for that time step.
           - Compute an error as: error = (S * _k.unsqueeze(-1)).sum(dim=-2)
           - Subtract that from _v and scale by beta to form an update term.
           - Update the memory S with an outer product of _k and the update.
           - Set the output at time i as: o_i = _q @ S (using an appropriate contraction).
    """
    orig_dtype = q.dtype
    b, h, l, d = q.shape
    # Convert to float for computation
    q, k, v, beta = map(lambda x: x.float(), [q, k, v, beta])
    # S: memory state, shape (B, H, D, D)
    S = torch.zeros(b, h, d, d, device=v.device, dtype=v.dtype)
    # Scale q
    # q = q * (d ** -0.5)
    
    # Make beta have shape (B, H, L, 1) if needed.
    if beta.ndim < v.ndim:
        beta = beta[..., None]
    
    o = torch.zeros_like(v)
    
    for i in range(l):
        _q = q[:, :, i]        # shape (B, H, D)
        _k = k[:, :, i]        # shape (B, H, D)
        _v = v[:, :, i].clone()  # shape (B, H, D)
        beta_i = beta[:, :, i]   # shape (B, H, 1)
        
        # Compute an error term from the current state S and the current key:
        error = (S * _k.unsqueeze(-1)).sum(dim=-2)  # shape (B, H, D)
        # Subtract the error from _v and scale by beta:
        update = (_v - error) * beta_i             # shape (B, H, D)
        # Update S: outer product between _k and update
        S = S + _k.unsqueeze(-1) * update.unsqueeze(-2)  # shape (B, H, D, D)
        # Compute output: multiply _q (B, H, D) with S (B, H, D, D) over the last dimension:
        o[:, :, i] = torch.einsum('bhd,bhdm->bhm', _q, S)
    
    if not output_final_state:
        S = None
    return o.to(orig_dtype), S

# Test dimensions
B = 8 #16 #1
H = 16 #8 #1
N = 128
D = 16 #64
beta_value = 0.01  # you can adjust beta

TESTNAME = sys.argv[1] if len(sys.argv) > 1 else 'randn_all'

if TESTNAME.startswith('ones'):
    q = (torch.ones((B, H, N, D), dtype=torch.bfloat16, device='cuda')/D).to(torch.float32)
    k = (torch.ones((B, H, N, D), dtype=torch.bfloat16, device='cuda')/D).to(torch.float32)
    v = (torch.ones((B, H, N, D), dtype=torch.bfloat16, device='cuda')/D).to(torch.float32)
elif TESTNAME.startswith('randn'):
    torch.manual_seed(42)
    q = (torch.randn((B, H, N, D), dtype=torch.bfloat16, device='cuda')/(D**0.5)).to(torch.float32)
    k = (torch.randn((B, H, N, D), dtype=torch.bfloat16, device='cuda')/(D**0.5)).to(torch.float32)
    print(k)
    v = (torch.randn((B, H, N, D), dtype=torch.bfloat16, device='cuda')/D).to(torch.float32)
else:
    print("Invalid test name")
    sys.exit(1)

# Create a beta tensor of shape (B, H, N)
beta = torch.full((B, H, N), beta_value, device=q.device, dtype=q.dtype)

# Compute delta attention output using our recurrence function.
o, s_new = delta_rule_recurrence(q, k, v, beta, initial_state=None, output_final_state=True)

# Flatten each tensor into a list of floats.
q_flat = q.flatten().cpu().numpy().tolist()
k_flat = k.flatten().cpu().numpy().tolist()
v_flat = v.flatten().cpu().numpy().tolist()
o_flat = o.flatten().cpu().numpy().tolist()

filename = f"delta_fwd_{B}x{H}x{N}x{D}.txt"
with open(filename, "w") as f:
    for val in trange(len(q_flat), desc="Writing Q"):
        f.write(f"{q_flat[val]} ")
    for val in trange(len(k_flat), desc="Writing K"):
        f.write(f"{k_flat[val]} ")
    for val in trange(len(v_flat), desc="Writing V"):
        f.write(f"{v_flat[val]} ")
    for val in trange(len(o_flat), desc="Writing O_ref"):
        f.write(f"{o_flat[val]} ")
print(f"Written delta attention reference data to {filename}")
