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

import torch

def chunk_delta_rule_forward(Q, K, V, beta, C, initial_state=None, output_final_state=True):
    """
    Delta rule forward pass with chunking, supporting batching and multi-head attention.

    Args:
        Q (torch.Tensor): Query tensor of shape [B, H, N, D]
        K (torch.Tensor): Key tensor of shape [B, H, N, D]
        V (torch.Tensor): Value tensor of shape [B, H, N, D]
        beta (torch.Tensor): Beta tensor of shape [B, H, N]
        C (int): Chunk size
        initial_state (torch.Tensor or None): Optional initial state of shape [B, H, D, D]
        output_final_state (bool): Whether to return the final state

    Returns:
        torch.Tensor: Output tensor of shape [B, H, N, D]
        torch.Tensor or None: Final state tensor of shape [B, H, D, D] if output_final_state is True
    """
    orig_dtype = Q.dtype
    B, H, N, D = Q.shape
    num_chunks = N // C

    # Reshape to chunked view: [B, H, num_chunks, C, D]
    Q_chunks = Q.view(B, H, num_chunks, C, D)
    K_chunks = K.view(B, H, num_chunks, C, D)
    V_chunks = V.view(B, H, num_chunks, C, D)
    beta_chunks = beta.view(B, H, num_chunks, C)

    # Broadcast beta for element-wise product
    K_beta = K_chunks * beta_chunks.unsqueeze(-1)
    V_beta = V_chunks * beta_chunks.unsqueeze(-1)

    # Build T matrix using vectorized forward substitution
    T = torch.ones(B, H, num_chunks, C, C, device=Q.device, dtype=Q.dtype)
    # T = -(K_beta @ K_chunks.transpose(-1, -2)) #.tril(-1)  # [B, H, num_chunks, C, C]
    print("T value:", T)
    # for i in range(1, C):
    #     T[:, :, :, i, :i] += (T[:, :, :, i, :, None] * T[:, :, :, :, :i]).sum(-2)

    # T += torch.eye(C, device=Q.device, dtype=Q.dtype).unsqueeze(0).unsqueeze(0).unsqueeze(0)

    # Compute intermediate W and U
    # W = T @ K_beta  # [B, H, num_chunks, C, D]
    W = torch.ones(B, H, num_chunks, C, D, device=Q.device, dtype=Q.dtype)  # Placeholder for W
    # U = T @ V_beta  # [B, H, num_chunks, C, D]
    U = torch.ones(B, H, num_chunks, C, D, device=Q.device, dtype=Q.dtype)  # Placeholder for U

    # Initialize state and output
    S = initial_state if initial_state is not None else torch.ones(B, H, D, D, device=Q.device, dtype=Q.dtype)
    O = torch.empty_like(V)  # [B, H, N, D]

    for i in range(num_chunks):
        q_i = Q_chunks[:, :, i]       # [B, H, C, D]
        k_i = K_chunks[:, :, i]       # [B, H, C, D]
        w_i = W[:, :, i]              # [B, H, C, D]
        u_i = U[:, :, i] - w_i @ S    # [B, H, C, D]
        o_inter = q_i @ S             # [B, H, C, D]
        A_i = (q_i @ k_i.transpose(-1, -2)).tril()  # [B, H, C, C]
        o_intra = A_i @ u_i           # [B, H, C, D]
        # S = S + k_i.transpose(-1, -2) @ u_i  # [B, H, D, D]
        O[:, :, i * C : (i + 1) * C] = o_intra + o_inter

    if not output_final_state:
        S = None

    return O.to(orig_dtype), S



# USE THIS FOR NOW
def delta_rule_recurrence_modified(q, k, v, beta, initial_state=None, output_final_state=True):
    orig_dtype = q.dtype
    b, h, l, d = q.shape  # d is both d_k and d_v in your kernel
    q, k, v, beta = map(lambda x: x.float(), [q, k, v, beta])
    o = torch.zeros_like(v)
    S = torch.zeros(b, h, d, d).to(v)
    q = q * (d ** -0.5)

    if beta.ndim < v.ndim:
        beta = beta[..., None]

    if initial_state is not None:
        S += initial_state

    for i in range(l):
        _q = q[:, :, i]         # [b, h, d]
        _k = k[:, :, i]         # [b, h, d]
        _v = v[:, :, i]         # [b, h, d]
        beta_i = beta[:, :, i]  # [b, h, 1]

        # Compute error = (S @ kᵀ) - v
        k_T = _k.unsqueeze(-1)  # [b, h, d, 1]
        pred = torch.matmul(S, k_T).squeeze(-1)  # [b, h, d]
        error = pred - _v

        # beta * error
        beta_error = beta_i * error

        # Δ = beta_error ⊗ k
        delta = torch.einsum('bhi,bhj->bhij', beta_error, _k)

        # S = S - Δ
        S = S - delta

        # o_i = S @ q
        q_col = _q.unsqueeze(-1)  # [b, h, d, 1]
        out = torch.matmul(S, q_col).squeeze(-1)
        o[:, :, i] = out

    S = None if not output_final_state else S
    return o.to(orig_dtype), S

# DOESN'T WORK
def delta_rule_recurrence_modified_fully_vectorized(q, k, v, beta, initial_state=None, output_final_state=True):
    orig_dtype = q.dtype
    b, h, l, d = q.shape  # d is both d_k and d_v in your kernel
    q, k, v, beta = map(lambda x: x.float(), [q, k, v, beta])
    q = q * (d ** -0.5)

    if beta.ndim < v.ndim:
        beta = beta[..., None]

    # Initialize outputs and state
    o = torch.zeros_like(v)
    
    # Create cumulative state tensor
    if initial_state is not None:
        S_0 = initial_state.clone()
    else:
        S_0 = torch.zeros(b, h, d, d).to(v)
    
    # Use torch.cumsum for running calculations
    # First prepare the delta matrices for all positions
    
    # For each position in the sequence:
    # 1. Compute the prediction based on the current state
    # 2. Calculate the error
    # 3. Update the state
    # 4. Generate the output
    
    # We'll use scan-like operations to do this efficiently
    
    # Initialize tensors to store intermediate results
    states = torch.zeros(b, h, l+1, d, d).to(v)
    states[:, :, 0] = S_0
    
    for i in range(l):
        # Get current key and value
        k_i = k[:, :, i]  # [b, h, d]
        v_i = v[:, :, i]  # [b, h, d]
        beta_i = beta[:, :, i]  # [b, h, 1]
        
        # Current state
        S_i = states[:, :, i]  # [b, h, d, d]
        
        # Prediction
        k_i_expanded = k_i.unsqueeze(-1)  # [b, h, d, 1]
        pred_i = torch.matmul(S_i, k_i_expanded).squeeze(-1)  # [b, h, d]
        
        # Error
        error_i = pred_i - v_i  # [b, h, d]
        
        # Delta
        delta_i = torch.einsum('bhi,bhj->bhij', beta_i * error_i, k_i)  # [b, h, d, d]
        
        # Update state
        states[:, :, i+1] = S_i - delta_i
        
        # Output
        q_i = q[:, :, i]  # [b, h, d]
        q_i_expanded = q_i.unsqueeze(-1)  # [b, h, d, 1]
        o[:, :, i] = torch.matmul(states[:, :, i], q_i_expanded).squeeze(-1)  # [b, h, d]
    
    S_final = None if not output_final_state else states[:, :, -1]
    return o.to(orig_dtype), S_final

def delta_rule_recurrence_blocked(q, k, v, beta, active_tiles=4, rows=16, output_final_state=True):
    orig_dtype = q.dtype
    b, h, l, d = q.shape
    q, k, v, beta = map(lambda x: x.float(), [q, k, v, beta])
    q = q * (d ** -0.5)

    o = torch.zeros_like(v)
    num_blocks = l // (active_tiles * rows)

    # Shared state buffer (rolling): [b, h, d, d] for each tile+1
    S_buf = [torch.zeros(b, h, d, d, dtype=torch.float32, device=q.device) for _ in range(active_tiles + 1)]

    for block in range(num_blocks):
        total_block_idx = (block * active_tiles) % (active_tiles + 1)
        for warpid in range(active_tiles):
            idx = block * active_tiles * rows + warpid * rows  # start index for this tile
            if idx >= l:
                continue

            q_tile = q[:, :, idx:idx+rows]       # [b, h, rows, d]
            k_tile = k[:, :, idx:idx+rows]
            v_tile = v[:, :, idx:idx+rows]
            beta_tile = beta[:, :, idx:idx+rows, None]

            S = S_buf[(total_block_idx + warpid) % (active_tiles + 1)].clone()

            for r in range(rows):
                _q = q_tile[:, :, r]
                _k = k_tile[:, :, r]
                _v = v_tile[:, :, r]
                beta_r = beta_tile[:, :, r]

                pred = torch.matmul(S, _k.unsqueeze(-1)).squeeze(-1)  # [b, h, d]
                error = pred - _v
                beta_error = beta_r * error
                delta = torch.einsum('bhi,bhj->bhij', beta_error, _k)
                S = S - delta

                out = torch.matmul(S, _q.unsqueeze(-1)).squeeze(-1)
                o[:, :, idx + r] = out

            # Save updated state to shared buffer
            S_buf[(total_block_idx + warpid + 1) % (active_tiles + 1)] = S

    return o.to(orig_dtype), S_buf


# Test dimensions
B = 8 #16 #1
H = 8 #8 #1
N = 128
D = 16 #64
ROWS = 16
beta_value = 0.01  # you can adjust beta

TESTNAME = sys.argv[1] if len(sys.argv) > 1 else 'randn_all'

if TESTNAME.startswith('ones'):
    q = (torch.ones((B, H, N, D), dtype=torch.bfloat16, device='cuda')/(D**0.5)).to(torch.float32)
    k = (torch.ones((B, H, N, D), dtype=torch.bfloat16, device='cuda')/(D**0.5)).to(torch.float32)
    v = (torch.ones((B, H, N, D), dtype=torch.bfloat16, device='cuda')/(D**0.5)).to(torch.float32)
elif TESTNAME.startswith('randn'):
    torch.manual_seed(42)
    q = (torch.randn((B, H, N, D), dtype=torch.bfloat16, device='cuda')/(D**0.5)).to(torch.float32)
    k = (torch.randn((B, H, N, D), dtype=torch.bfloat16, device='cuda')/(D**0.5)).to(torch.float32)
    v = (torch.randn((B, H, N, D), dtype=torch.bfloat16, device='cuda')/(D**0.5)).to(torch.float32)
else:
    print("Invalid test name")
    sys.exit(1)

# Create a beta tensor of shape (B, H, N)
beta = torch.full((B, H, N), beta_value, device=q.device, dtype=q.dtype)

# Compute delta attention output using our recurrence function.
#o, s_new = delta_rule_recurrence_modified(q, k, v, beta, output_final_state=True)
#o, s_new = delta_rule_recurrence(q, k, v, beta, output_final_state=True)
o, s_nw = chunk_delta_rule_forward(q, k, v, beta, ROWS, initial_state=None, output_final_state=True)

# Flatten each tensor into a list of floats.
q_flat = q.flatten().cpu().numpy().tolist()
k_flat = k.flatten().cpu().numpy().tolist()
v_flat = v.flatten().cpu().numpy().tolist()
o_flat = o.flatten().cpu().numpy().tolist()

filename = f"debug_delta_fwd_{B}x{H}x{N}x{D}.txt"
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
