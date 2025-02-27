import torch
from tqdm import trange
import numpy as np
import sys
import math

# Read command-line arguments: sequence length, head dimension, # query heads, # key/value heads
N = int(sys.argv[1])
D = int(sys.argv[2])
H_QO = int(sys.argv[3])
H_KV = int(sys.argv[4])
causal = False  # set to True if you want to later incorporate a causal mask (the recurrence itself is causal)

B = 1  # single batch for fast file I/O

# For reproducibility
torch.manual_seed(42)

# Generate random tensors (using bfloat16) for Q, K, V.
# Q has shape (B, H_QO, N, D) and K,V have shape (B, H_KV, N, D).
q = torch.randn((B, H_QO, N, D), dtype=torch.bfloat16, device='cuda').requires_grad_()
k = torch.randn((B, H_KV, N, D), dtype=torch.bfloat16, device='cuda').requires_grad_()
v = torch.randn((B, H_KV, N, D), dtype=torch.bfloat16, device='cuda').requires_grad_()
grad_output = torch.randn((B, H_QO, N, D), dtype=torch.bfloat16, device='cuda')

# Helper: repeat keys/values along the head dimension to match Q if needed.
def repeat_kv(x: torch.Tensor, n_rep: int) -> torch.Tensor:
    # x is expected to have shape (B, seq_len, n_kv_heads, head_dim)
    bs, slen, n_kv_heads, head_dim = x.shape
    if n_rep == 1:
        return x
    return (
        x[:, :, :, None, :]
        .expand(bs, slen, n_kv_heads, n_rep, head_dim)
        .reshape(bs, slen, n_kv_heads * n_rep, head_dim)
    )

# DeltaNet linear attention implemented via sequential recurrence.
# For each time step t, we update the state S as:
#    S_t = S_{t-1} + β * (v_t - (S_{t-1} @ k_t)) ⊗ k_t,
# and compute the output as o_t = S_t @ q_t.
def deltanet_linear_attention(q, k, v, beta=0.5):
    # q: (B, H_QO, N, D)
    # k,v: (B, H_KV, N, D) --- if H_QO != H_KV, we will repeat them.
    B, H_QO, N, D = q.shape
    B_k, H_KV, N_k, D_k = k.shape
    assert B == B_k and N == N_k and D == D_k, "Dimension mismatch"

    # Repeat k and v to match query head count if necessary.
    n_rep = H_QO // H_KV
    if n_rep > 1:
        # Permute to (B, N, H_KV, D) and repeat, then permute back.
        k_rep = repeat_kv(k.permute(0, 2, 1, 3), n_rep).permute(0, 2, 1, 3)
        v_rep = repeat_kv(v.permute(0, 2, 1, 3), n_rep).permute(0, 2, 1, 3)
    else:
        k_rep = k
        v_rep = v

    # Initialize state S to zeros.
    # S has shape (B, H_QO, D, D) per head.
    S = torch.zeros((B, H_QO, D, D), dtype=q.dtype, device=q.device)
    outputs = []
    
    # Process each time step sequentially.
    for t in range(N):
        # Extract time-step t vectors for each head.
        # k_t, v_t, q_t have shape (B, H_QO, D)
        k_t = k_rep[:, :, t, :]
        v_t = v_rep[:, :, t, :]
        q_t = q[:, :, t, :]

        # Compute current prediction: pred = S @ k_t, shape (B, H_QO, D)
        # Here we use unsqueeze/squeeze to perform batched matrix multiplication.
        pred = torch.matmul(S, k_t.unsqueeze(-1)).squeeze(-1)

        # Delta update: compute pseudo-value u_t = β * (v_t - pred)
        u_t = beta * (v_t - pred)

        # Update state: S = S + outer(u_t, k_t)
        # Outer product computed per batch and head.
        delta = torch.einsum("bhd,bhe->bhde", u_t, k_t)
        S = S + delta

        # Compute output for time step t: o_t = S @ q_t, shape (B, H_QO, D)
        o_t = torch.matmul(S, q_t.unsqueeze(-1)).squeeze(-1)
        outputs.append(o_t)
    
    # Stack outputs along the time dimension -> shape (B, H_QO, N, D)
    output = torch.stack(outputs, dim=2)
    return output, S

# Run DeltaNet forward pass.
# Here, beta is set to 0.5 for all time steps; in practice beta_t might be computed per token.
o_delta, S_final = deltanet_linear_attention(q, k, v, beta=0.5)

# Now perform backward pass.
o_delta.backward(grad_output)

# Extract gradients.
q_grad = q.grad
k_grad = k.grad
v_grad = v.grad

# For diagnostics, compute a d_vec similar to the original script:
# d_vec = elementwise product of o_delta and grad_output, summed over the head dimension.
d_vec = (o_delta.to(torch.float32) * grad_output.to(torch.float32)).sum(dim=-1, keepdim=True)

print("--------------------------------------")
print("Q shape: ", q.shape)
print("K shape: ", k.shape)
print("V shape: ", v.shape)
print("O (DeltaNet) shape: ", o_delta.shape)
print("Q grad shape: ", q_grad.shape)
print("K grad shape: ", k_grad.shape)
print("V grad shape: ", v_grad.shape)
print("D shape: ", d_vec.shape)
print("--------------------------------------")

# Print average magnitudes (and 1/100 of them) for debugging.
print(f'Average magnitude of OUTPUT tensor: {o_delta.abs().mean()}')
print(f'1/100 magnitude of OUTPUT tensor:   {o_delta.abs().mean()/100}')
print(f'Average magnitude of Q_GRAD tensor: {q_grad.abs().mean()}')
print(f'1/100 magnitude of Q_GRAD tensor:   {q_grad.abs().mean()/100}')
print(f'Average magnitude of K_GRAD tensor: {k_grad.abs().mean()}')
print(f'1/100 magnitude of K_GRAD tensor:   {k_grad.abs().mean()/100}')
print(f'Average magnitude of V_GRAD tensor: {v_grad.abs().mean()}')
print(f'1/100 magnitude of V_GRAD tensor:   {v_grad.abs().mean()/100}')
print(f'Average magnitude of D tensor:      {d_vec.abs().mean()}')
print(f'1/100 magnitude of D tensor:        {d_vec.abs().mean()/100}')
print("--------------------------------------")

# Construct filename based on input parameters.
filename = f"randn_{N}N_{D}D_{H_QO}QO_{H_KV}KV"
if causal:
    filename += "_causal"
if H_QO != H_KV:
    filename += "_gqa"
filename += "_deltanet.txt"

# Write flattened arrays to file.
with open(filename, 'w') as f:
    # Convert tensors to float32 and flatten.
    qf = q.to(torch.float32).flatten().detach().cpu().numpy()
    kf = k.to(torch.float32).flatten().detach().cpu().numpy()
    vf = v.to(torch.float32).flatten().detach().cpu().numpy()
    of = o_delta.to(torch.float32).flatten().detach().cpu().numpy()
    og_f = grad_output.to(torch.float32).flatten().detach().cpu().numpy()
    d_vecf = d_vec.to(torch.float32).flatten().detach().cpu().numpy()
    qg_f = q_grad.to(torch.float32).flatten().detach().cpu().numpy()
    kg_f = k_grad.to(torch.float32).flatten().detach().cpu().numpy()
    vg_f = v_grad.to(torch.float32).flatten().detach().cpu().numpy()

    for arr in [qf, kf, vf, of, d_vecf, og_f, qg_f, kg_f, vg_f]:
        for val in trange(arr.size, desc="Writing array"):
            f.write(f"{float(arr[val])} ")
