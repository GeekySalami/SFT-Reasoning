"""
Skeleton for the custom flash-attention implementation.

Run the harness against THIS FILE FIRST, before writing any Triton code,
to confirm the harness itself, your torch/ROCm install, and the interface
all work end-to-end:

    python flash_attn_harness.py --impl my_kernel:flash_attn_fwd --quick

This stub just calls SDPA internally, so it MUST pass trivially. If it
doesn't pass, the bug is in your environment or the harness, not in a
kernel you haven't written yet -- fix that first.

Once this trivially passes, replace the *inside* of flash_attn_fwd with
a call into your real Triton kernel (forward) and wire up a matching
torch.autograd.Function with a custom backward calling your Triton
backward kernel. The harness doesn't care how you implement it, only
that flash_attn_fwd(q, k, v, causal, sm_scale) -> Tensor is correct
and differentiable.
"""

import torch
import torch.nn.functional as F


def flash_attn_fwd(q, k, v, causal: bool, sm_scale: float):
    """
    Placeholder -- delegates to SDPA so the harness has something to run
    against immediately. Replace this with your Triton kernel call once
    you're ready; keep the signature identical.
    """
    return F.scaled_dot_product_attention(q, k, v, is_causal=causal, scale=sm_scale)