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
causal = False  # set to True if you want to incorporate causal mask

B = 1  # single batch for fast file I/O

# For reproducibility
torch.manual_seed(42)

# Generate random tensors
q = torch.randn((B, H_QO, N, D), dtype=torch.float32, device='cuda').requires_grad_()
k = torch.randn((B, H_KV, N, D), dtype=torch.float32, device='cuda').requires_grad_()
v = torch.randn((B, H_KV, N, D), dtype=torch.float32, device='cuda').requires_grad_()
grad_output = torch.randn((B, H_QO, N, D), dtype=torch.float32, device='cuda')

# Helper: repeat keys/values along the head dimension to match Q if needed (GQA support)
def repeat_kv(x: torch.Tensor, n_rep: int) -> torch.Tensor:
    """Repeat key/value heads to match query head count for GQA"""
    B, H, N, D = x.shape
    if n_rep == 1:
        return x
    return (
        x[:, :, None, :, :]
        .expand(B, H, n_rep, N, D)
        .reshape(B, H * n_rep, N, D)
    )

def deltanet_linear_attention(q, k, v, beta=0.1, eps=1e-6):
    """
    Implements the delta rule recurrence for linear attention.
    Follows the logic from the reference implementation but with added
    numerical stability measures.
    
    Args:
        q: Queries tensor of shape (B, H_QO, N, D)
        k: Keys tensor of shape (B, H_KV, N, D)
        v: Values tensor of shape (B, H_KV, N, D)
        beta: Learning rate for the delta rule updates (scalar or tensor)
        eps: Small constant for numerical stability
        
    Returns:
        Tuple of (outputs, final_state)
    """
    # Save original dtype for consistent outputs
    orig_dtype = q.dtype
    
    # Get dimensions
    B, H_QO, N, D = q.shape
    B_k, H_KV, N_k, D_k = k.shape
    assert B == B_k and N == N_k and D == D_k, "Dimension mismatch"
    
    # Convert all tensors to float32 for stability during computation
    q, k, v = map(lambda x: x.float(), [q, k, v])
    
    # Scale q by 1/sqrt(d_k) for stable attention
    q = q * (D ** -0.5)
    
    # Repeat k and v to match query head count if necessary (for GQA)
    n_rep = H_QO // H_KV
    if n_rep > 1:
        k = repeat_kv(k, n_rep)
        v = repeat_kv(v, n_rep)
    
    # Convert beta to tensor if it's a scalar
    if isinstance(beta, float):
        beta = torch.full((B, H_QO, N), beta, dtype=torch.float32, device=q.device)
    
    # Ensure beta has correct dimensions for broadcasting
    if beta.ndim < v.ndim:
        beta = beta[..., None]
    
    # Initialize state and output
    output = torch.zeros_like(v, dtype=torch.float32)
    S = torch.zeros(B, H_QO, D, D, dtype=torch.float32, device=q.device)
    
    # Process sequence position by position (recurrence)
    for i in range(N):
        # Extract vectors for current position
        k_i = k[:, :, i]  # (B, H_QO, D)
        q_i = q[:, :, i]  # (B, H_QO, D)
        v_i = v[:, :, i].clone()  # (B, H_QO, D)
        beta_i = beta[:, :, i]  # (B, H_QO, 1)
        
        # Compute the prediction via S @ k_i
        # Reshape for efficient matrix multiplication
        k_i_expanded = k_i.unsqueeze(-1)  # (B, H_QO, D, 1)
        pred = torch.matmul(S, k_i_expanded).squeeze(-1)  # (B, H_QO, D)
        
        # Compute prediction error and scale by beta
        delta = v_i - pred  # (B, H_QO, D)
        delta = delta * beta_i  # (B, H_QO, D)
        
        # Update state with outer product: S += delta ⊗ k_i^T
        k_i_transposed = k_i.unsqueeze(-2)  # (B, H_QO, 1, D)
        delta_expanded = delta.unsqueeze(-1)  # (B, H_QO, D, 1)
        S = S + torch.matmul(delta_expanded, k_i_transposed)  # (B, H_QO, D, D)
        
        # Compute output for this position: o_i = S @ q_i
        q_i_expanded = q_i.unsqueeze(-1)  # (B, H_QO, D, 1)
        output[:, :, i] = torch.matmul(S, q_i_expanded).squeeze(-1)  # (B, H_QO, D)
    
    # Return to original dtype for consistency
    return output.to(orig_dtype), S

# Run DeltaNet forward pass with a smaller beta for stability
beta_value = 0.05  # Using a smaller beta value for numerical stability
o_delta, S_final = deltanet_linear_attention(q, k, v, beta=beta_value)

# Ensure no NaNs in output
assert not torch.isnan(o_delta).any(), "NaN detected in output!"

# Perform backward pass
o_delta.backward(grad_output)

# Extract gradients
q_grad = q.grad
k_grad = k.grad
v_grad = v.grad

# For diagnostics, compute a d_vec (attention sensitivity metric)
d_vec = (o_delta * grad_output).sum(dim=-1, keepdim=True)

print("--------------------------------------")
print("Q shape: ", q.shape)
print("K shape: ", k.shape)
print("V shape: ", v.shape)
print("O (DeltaNet) shape: ", o_delta.shape)
print("Q grad shape: ", q_grad.shape)
print("K grad shape: ", k_grad.shape)
print("V grad shape: ", v_grad.shape)
print("D shape: ", d_vec.shape)
print("S final shape: ", S_final.shape)
print("--------------------------------------")

# Check for NaNs in gradients
for name, tensor in [("q_grad", q_grad), ("k_grad", k_grad), ("v_grad", v_grad)]:
    if torch.isnan(tensor).any():
        print(f"WARNING: NaNs detected in {name}")
    else:
        print(f"{name} is clean (no NaNs)")

# Print average magnitudes for debugging
print(f'Average magnitude of OUTPUT tensor: {o_delta.abs().mean().item()}')
print(f'1/100 magnitude of OUTPUT tensor:   {(o_delta.abs().mean()/100).item()}')
print(f'Average magnitude of Q_GRAD tensor: {q_grad.abs().mean().item()}')
print(f'1/100 magnitude of Q_GRAD tensor:   {(q_grad.abs().mean()/100).item()}')
print(f'Average magnitude of K_GRAD tensor: {k_grad.abs().mean().item()}')
print(f'1/100 magnitude of K_GRAD tensor:   {(k_grad.abs().mean()/100).item()}')
print(f'Average magnitude of V_GRAD tensor: {v_grad.abs().mean().item()}')
print(f'1/100 magnitude of V_GRAD tensor:   {(v_grad.abs().mean()/100).item()}')
print(f'Average magnitude of D tensor:      {d_vec.abs().mean().item()}')
print(f'1/100 magnitude of D tensor:        {(d_vec.abs().mean()/100).item()}')
print("--------------------------------------")

# Construct filename based on input parameters
filename = f"randn_{N}N_{D}D_{H_QO}QO_{H_KV}KV"
if causal:
    filename += "_causal"
if H_QO != H_KV:
    filename += "_gqa"
filename += "_deltanet.txt"

# Write tensors to file
with open(filename, 'w') as f:
    tensors = [
        q.detach().cpu().float(),
        k.detach().cpu().float(),
        v.detach().cpu().float(),
        o_delta.detach().cpu().float(),
        grad_output.detach().cpu().float(),
        d_vec.detach().cpu().float(),
        q_grad.detach().cpu().float(),
        k_grad.detach().cpu().float(),
        v_grad.detach().cpu().float()
    ]
    
    for tensor in tensors:
        for val in tensor.flatten().numpy():
            f.write(f"{repr(float(val))} ")