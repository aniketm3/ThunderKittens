import torch
import math

def rope_apply(x, rope_freqs):
    """
    x: shape [B, T, n_heads, d_rope]
    rope_freqs: shape [T, d_rope/2] as complex exponentials
    Returns same shape as x, with RoPE applied on last dimension.
    """
    B, T, H, D = x.shape
    assert D % 2 == 0, "RoPE dimension must be even to form complex pairs."

    # Convert real -> complex
    x_2 = x.float().reshape(B, T, H, D // 2, 2)  # separate real/imag
    x_complex = torch.view_as_complex(x_2)

    # Expand rope_freqs from [T, D/2] to [1, T, 1, D/2]
    rope_freqs = rope_freqs.view(1, T, 1, D // 2).to(x.device).to(x.dtype)

    # Multiply
    x_rot = x_complex * rope_freqs

    # Convert complex -> real
    out = torch.view_as_real(x_rot).reshape(B, T, H, D)
    return out.to(x.dtype)

def mla_forward(
    h,                # [B, T, d]
    W_D_KV,           # [d_c, d]      ; eq.1
    W_U_K,            # [n_heads*d_h, d_c]
    W_U_V,            # [n_heads*d_h, d_c]
    W_K_R,            # [n_heads*d_hR, d]  ; eq.3
    W_D_Q,            # [d_cQ, d]     ; eq.6
    W_U_Q,            # [n_heads*d_h, d_cQ]
    W_Q_R,            # [n_heads*d_hR, d_cQ]
    W_O,              # [d, n_heads*(d_h)],  (since values are d_h each, no rope part)
    rope_freqs,       # [T, d_hR] as complex exponentials (half real/imag)
    n_heads,
    d_h, d_hR,        # dimension of compressed part per head, dimension of rope part
    causal_mask=None
):
    """
    Implementation of Eqs. (1)–(11). 
    We assume:
      - h in shape [B, T, d].
      - The "compressed" part is n_heads*d_h
      - The "rope" part is n_heads*d_hR
      - We do full-sequence attention (no caching).
      - W_O yields a final [B, T, d].
    """
    B, T, d = h.shape

    # ------------------------------------------------------
    # 1) c^K_V = W^D_{K,V}  h_t
    #    shape => [B, T, d_c]
    # ------------------------------------------------------
    cKV = (h @ W_D_KV.T)  # eq.1

    # ------------------------------------------------------
    # 2) k^C = W^U_K * c^K_V
    #    shape => [B, T, n_heads*d_h]
    # ------------------------------------------------------
    kC = (cKV @ W_U_K.T)  # eq.2
    # reshape => [B, T, n_heads, d_h]
    kC = kC.reshape(B, T, n_heads, d_h)

    # (optionally: v^C = W^U_V * c^K_V => [B, T, n_heads, d_h])
    vC = (cKV @ W_U_V.T)  # eq.5
    vC = vC.reshape(B, T, n_heads, d_h)

    # ------------------------------------------------------
    # 3) k^R = RoPE( W^K_R * h_t )
    #    shape => [B, T, n_heads*d_hR]
    # ------------------------------------------------------
    kR_raw = h @ W_K_R.T  # eq.3 => shape [B, T, n_heads*d_hR]
    kR = kR_raw.reshape(B, T, n_heads, d_hR)
    kR = rope_apply(kR, rope_freqs)  # eq.3

    # Then k = concat( k^C, k^R ) => [B, T, n_heads, d_h + d_hR]
    kFull = torch.cat([kC, kR], dim=-1)  # eq.4

    # ------------------------------------------------------
    # 6) c^Q = W^D_Q * h
    # ------------------------------------------------------
    cQ = (h @ W_D_Q.T)  # shape [B, T, d_cQ]

    # ------------------------------------------------------
    # 7) q^C = W^U_Q * c^Q => [B, T, n_heads*d_h]
    # ------------------------------------------------------
    qC = (cQ @ W_U_Q.T).reshape(B, T, n_heads, d_h)  # eq.7

    # ------------------------------------------------------
    # 8) q^R = RoPE( W^Q_R * c^Q )
    #    => [B, T, n_heads*d_hR]
    # ------------------------------------------------------
    qR_raw = (cQ @ W_Q_R.T)  # shape [B, T, n_heads*d_hR]
    qR = qR_raw.reshape(B, T, n_heads, d_hR)
    qR = rope_apply(qR, rope_freqs)

    # 9) q = concat( q^C, q^R ) => [B, T, n_heads, d_h + d_hR]
    qFull = torch.cat([qC, qR], dim=-1)

    # ------------------------------------------------------
    # Dot-product attention
    # shape => [B, n_heads, T, (d_h + d_hR)] x [B, n_heads, T, (d_h + d_hR)]
    # => scores => [B, n_heads, T, T]
    # eq.(10) divides by sqrt(d_h + d_hR)
    # ------------------------------------------------------
    scale = 1.0 / math.sqrt(d_h + d_hR)
    scores = torch.einsum("bthd,bThd->bhtT", qFull, kFull) * scale

    # apply causal mask if provided
    if causal_mask is not None:
        scores = scores + causal_mask  # shape broadcast => [B, n_heads, T, T]

    attn_probs = torch.softmax(scores, dim=-1)  # eq.(10)

    # Then multiply by v^C => shape => [B, n_heads, T, d_h]
    # eq.(10) uses only v^C for the final output
    out_heads = torch.einsum("bhtT,bThd->bthd", attn_probs, vC)

    # ------------------------------------------------------
    # 11) W^O * concat over heads => shape => [B, T, d]
    #    out_heads => [B, T, n_heads, d_h] => flatten -> [B, T, n_heads*d_h]
    # ------------------------------------------------------
    out_heads_2d = out_heads.reshape(B, T, n_heads*d_h)
    output = torch.einsum("btd,xd->btx", out_heads_2d, W_O)  # eq.(11)

    # shape => [B, T, d]
    return output

# ------------------------------------------------------------------
# Example usage
if __name__ == "__main__":
    torch.manual_seed(42)
    B, T, d = 2, 8, 16   # small example
    n_heads = 2
    d_h = 4     # compressed dimension per head (the "C" portion)
    d_hR = 2    # rope dimension per head (the "R" portion)
    d_c = 6     # dimension for c^K_V
    d_cQ = 6    # dimension for c^Q

    device = "cuda"
    dtype = torch.float32   # or bfloat16 if you prefer
    h = torch.randn(B, T, d, device=device, dtype=dtype)

    # Make random projection matrices consistent with the shapes above
    W_D_KV = torch.randn(d_c, d, device=device, dtype=dtype)       # eq.1
    W_U_K  = torch.randn(n_heads*d_h, d_c, device=device, dtype=dtype) # eq.2
    W_U_V  = torch.randn(n_heads*d_h, d_c, device=device, dtype=dtype) # eq.5
    W_K_R  = torch.randn(n_heads*d_hR, d, device=device, dtype=dtype)  # eq.3

    W_D_Q  = torch.randn(d_cQ, d, device=device, dtype=dtype)      # eq.6
    W_U_Q  = torch.randn(n_heads*d_h, d_cQ, device=device, dtype=dtype) # eq.7
    W_Q_R  = torch.randn(n_heads*d_hR, d_cQ, device=device, dtype=dtype)# eq.8

    # eq.(11) => final output [d, n_heads*d_h] or [d, n_heads*(d_h + d_hR)] 
    #   but the paper says the final V is only d_h, not d_h + d_hR
    W_O = torch.randn(d, n_heads*d_h, device=device, dtype=dtype)

    # Precompute RoPE frequencies => shape [T, d_hR], viewed as complex 
    # For simplicity, produce [T, d_hR/2] as complex exponentials. 
    # We'll do a naive approach:
    def build_rope_freqs(T, d_hR):
        assert d_hR % 2 == 0, "rope dimension must be even"
        base = 10000.0
        half = d_hR // 2
        freq = 1.0 / (base ** (torch.arange(0, d_hR, 2, device=device, dtype=dtype)/d_hR))
        t_ = torch.arange(T, device=device, dtype=dtype)
        freq = torch.outer(t_, freq) # => [T, half]
        # Convert to complex exponent
        return torch.polar(torch.ones_like(freq), freq) # shape [T, half] complex

    rope_freqs = build_rope_freqs(T, n_heads*d_hR // n_heads)  
    # Actually, we only need d_hR per head, but this is enough to illustrate.

    # Optional causal mask => shape [T, T] or broadcast to [B, n_heads, T, T]
    causal_mask = torch.full((T, T), float("-inf"), device=device)
    causal_mask = torch.triu(causal_mask, diagonal=1)  # upper tri = -inf
    # expand to [1, n_heads, T, T], but we can rely on broadcast
    causal_mask = causal_mask.unsqueeze(0).unsqueeze(0)

    out = mla_forward(
        h, W_D_KV, W_U_K, W_U_V, W_K_R,
        W_D_Q, W_U_Q, W_Q_R, W_O,
        rope_freqs,
        n_heads, d_h, d_hR,
        causal_mask=causal_mask
    )
    print("Output shape:", out.shape)  # => [B, T, d]
