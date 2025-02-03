import torch
import sys
import math

# ------------------------------------------------------------------------------
# Parse Command-Line Arguments
# ------------------------------------------------------------------------------
if len(sys.argv) < 5:
    print("Usage: python gentests_mla.py <N> <D> <H_QO> <H_KV> [causal=0|1]")
    sys.exit(1)

B = 1                # Batch size fixed at 1
N = int(sys.argv[1]) # Sequence length
D = int(sys.argv[2]) # Model dimension
H_QO = int(sys.argv[3])  # Number of query/output heads
H_KV = int(sys.argv[4])  # Number of key/value heads

causal = False
if len(sys.argv) > 5 and sys.argv[5] == "1":
    causal = True

torch.random.manual_seed(42)
device = "cuda"
torch.set_default_device(device)
torch.set_default_dtype(torch.bfloat16)

# ------------------------------------------------------------------------------
# Create Random Tensors (Q, K, V) + grad_output
# Shapes: Q => [B, H_QO, N, D], K => [B, H_KV, N, D], V => [B, H_KV, N, D]
# ------------------------------------------------------------------------------
q = torch.randn((B, H_QO, N, D), dtype=torch.bfloat16, device=device, requires_grad=True)
k = torch.randn((B, H_KV, N, D), dtype=torch.bfloat16, device=device, requires_grad=True)
v = torch.randn((B, H_KV, N, D), dtype=torch.bfloat16, device=device, requires_grad=True)

grad_output = torch.randn((B, H_QO, N, D), dtype=torch.bfloat16, device=device)

# ------------------------------------------------------------------------------
# Simple RoPE Utilities
# ------------------------------------------------------------------------------
def precompute_freqs_cis(seq_len: int, dim: int):
    """
    Compute rotary positional frequencies of size [seq_len, dim//2],
    then convert them to complex exponentials for use as 'freqs_cis'.
    """
    base = 10000.0
    # We assume dim is even so (dim % 2 == 0).
    # freq positions go over half of dim (the real/imag pairs).
    half_dim = dim // 2
    freqs = 1.0 / (base ** (torch.arange(0, dim, 2, dtype=torch.float32) / dim))
    t = torch.arange(seq_len, dtype=torch.float32, device=device)
    freqs = torch.outer(t, freqs)  # shape [seq_len, half_dim]
    freqs_cis = torch.polar(torch.ones_like(freqs), freqs)  # convert magnitude/phase -> complex
    return freqs_cis.to(torch.bfloat16)

def apply_rope(x: torch.Tensor, freqs_cis: torch.Tensor):
    """
    Apply rotary embeddings to x.
    - x is shape [B, H, N, D]
    - We'll interpret the last dimension D as D/2 complex pairs.
    """
    B_, H_, N_, D_ = x.shape
    assert D_ % 2 == 0, "D must be even to form complex pairs."
    # Convert x to complex: shape => [B, H, N, D_/2, 2] -> complex -> [B, H, N, D_/2]
    x_float = x.float().reshape(B_, H_, N_, D_ // 2, 2)
    x_complex = torch.view_as_complex(x_float)

    # freqs_cis has shape [N, D_/2], expand to [1, 1, N, D_/2]
    freqs_cis = freqs_cis.view(1, 1, N_, D_ // 2).to(x.device)

    # Multiply
    x_rotated = x_complex * freqs_cis  # shape [B, H, N, D_/2] (complex)

    # Convert back to real
    out = torch.view_as_real(x_rotated).reshape(B_, H_, N_, D_)
    return out.to(x.dtype)

# ------------------------------------------------------------------------------
# Precompute Frequencies & Possibly Causal Mask
# ------------------------------------------------------------------------------
freqs_cis = precompute_freqs_cis(N, D //2) # linet o debug
mask = None
if causal:
    # shape [N, N], upper triangle -> -inf
    mask = torch.full((N, N), float('-inf'), device=device, dtype=torch.bfloat16)
    mask = torch.triu(mask, diagonal=1)  # causal mask

# ------------------------------------------------------------------------------
# Apply RoPE to Q and K
# ------------------------------------------------------------------------------
q_rope = apply_rope(q, freqs_cis)
k_rope = apply_rope(k, freqs_cis)

# ------------------------------------------------------------------------------
# Scale Q for stability (like 1/sqrt(D))
# ------------------------------------------------------------------------------
q_scaled = q_rope * (1.0 / math.sqrt(D))

# ------------------------------------------------------------------------------
# Compute Attention Scores
# q_scaled: [B, H_QO, N, D], k_rope: [B, H_KV, N, D]
# We'll do a simple matmul: (q_scaled @ k_rope^T) => [B, H_QO, N, N]
# (If H_QO != H_KV, you'd normally replicate or otherwise handle K/V, 
#  but here we just do a naive approach that requires H_QO == H_KV or 
#  it won't broadcast properly.)
# ------------------------------------------------------------------------------
scores = torch.matmul(q_scaled, k_rope.transpose(2, 3))  # shape [B, H_QO, N, N]

# Add causal mask if needed
if causal:
    scores = scores + mask  # broadcast -> shape [B, H_QO, N, N]

# softmax over last dim
attn_probs = torch.nn.functional.softmax(scores, dim=-1)

# ------------------------------------------------------------------------------
# Final Attention Output
# shape: [B, H_QO, N, D]
# ------------------------------------------------------------------------------
output = torch.matmul(attn_probs, v)

# ------------------------------------------------------------------------------
# Backward Pass
# ------------------------------------------------------------------------------
output.backward(grad_output)

q_grad = q.grad
k_grad = k.grad
v_grad = v.grad

# ------------------------------------------------------------------------------
# Print Debug Info
# ------------------------------------------------------------------------------
print("--------------------------------------")
print("Q shape:       ", q.shape)
print("K shape:       ", k.shape)
print("V shape:       ", v.shape)
print("Output shape:  ", output.shape)
print("Q grad shape:  ", q_grad.shape)
print("K grad shape:  ", k_grad.shape)
print("V grad shape:  ", v_grad.shape)
print("--------------------------------------")

print("Average magnitude of OUTPUT:", output.abs().mean())
print("Average magnitude of Q_GRAD:", q_grad.abs().mean())
print("Average magnitude of K_GRAD:", k_grad.abs().mean())
print("Average magnitude of V_GRAD:", v_grad.abs().mean())
print("--------------------------------------")

# ------------------------------------------------------------------------------
# Write Out Data to a File
# ------------------------------------------------------------------------------
filename = f"randn_{N}N_{D}D_{H_QO}QO_{H_KV}KV_mla"
if causal:
    filename += "_causal"
filename += ".txt"

with open(filename, 'w') as f:
    # Flatten everything to float32 on CPU
    qf = q.detach().cpu().float().flatten().numpy()
    kf = k.detach().cpu().float().flatten().numpy()
    vf = v.detach().cpu().float().flatten().numpy()
    of = output.detach().cpu().float().flatten().numpy()

    # Attn probabilities, grads, etc. if you want them:
    af = attn_probs.detach().cpu().float().flatten().numpy()
    qgf = q_grad.detach().cpu().float().flatten().numpy()
    kgf = k_grad.detach().cpu().float().flatten().numpy()
    vgf = v_grad.detach().cpu().float().flatten().numpy()

    # Write in a chosen order (Q, K, V, output, attention probs, Q_grad, K_grad, V_grad)
    for val in qf:
        f.write(repr(float(val)) + ' ')
    for val in kf:
        f.write(repr(float(val)) + ' ')
    for val in vf:
        f.write(repr(float(val)) + ' ')
    for val in of:
        f.write(repr(float(val)) + ' ')
    for val in af:
        f.write(repr(float(val)) + ' ')
    for val in qgf:
        f.write(repr(float(val)) + ' ')
    for val in kgf:
        f.write(repr(float(val)) + ' ')
    for val in vgf:
        f.write(repr(float(val)) + ' ')

print(f"Done. Test data written to: {filename}")