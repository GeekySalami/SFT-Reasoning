"""
Correctness harness for a custom (Triton/ROCm) flash-attention kernel.

Goal: before trusting the kernel in real training, prove numerically that
it matches PyTorch's SDPA reference on forward output AND on gradients
w.r.t. Q, K, V. This is the thing that turns "I think the algorithm is
right" into "I checked, and it is."

Usage:
    python flash_attn_harness.py --impl your_kernel_module:flash_attn_fwd

Your kernel function must have signature:
    out = flash_attn_fwd(q, k, v, causal: bool, sm_scale: float) -> Tensor

and must be differentiable via autograd (either a torch.autograd.Function
with a proper backward, or built from autograd-visible ops).

Design notes:
  - We test multiple (batch, heads, seq_len, head_dim) shapes, including
    non-power-of-2 seq_len (to catch tiling/boundary bugs) and seq_len
    smaller than / larger than typical block sizes.
  - We test both causal and non-causal.
  - We check forward AND backward (dQ, dK, dV) -- backward is where most
    real bugs live (online-softmax rescaling, recomputation in bwd).
  - We run in bf16 (matching your training config) AND fp32 (to separate
    "algorithm is wrong" from "bf16 precision is just lossy").
  - Tolerances are looser for bf16 than fp32, deliberately -- bf16 has
    ~3 decimal digits of precision, so exact match is not the bar.
"""

import argparse
import importlib
import sys
import torch
import torch.nn.functional as F


def get_device():
    if torch.cuda.is_available():
        return "cuda"  # ROCm reports itself as "cuda" to torch
    raise RuntimeError(
        "No GPU visible to torch. Run this on your iGPU machine with "
        "ROCm-enabled torch, not in a CPU-only sandbox."
    )


def reference_attention(q, k, v, causal, sm_scale):
    """PyTorch SDPA reference. This is ground truth."""
    return F.scaled_dot_product_attention(
        q, k, v, is_causal=causal, scale=sm_scale
    )


def make_inputs(batch, heads, seq_len, head_dim, dtype, device, seed):
    torch.manual_seed(seed)
    shape = (batch, heads, seq_len, head_dim)
    # requires_grad=True so autograd can flow through both the custom
    # kernel and the reference implementation for the backward check.
    q = torch.randn(shape, dtype=dtype, device=device, requires_grad=True)
    k = torch.randn(shape, dtype=dtype, device=device, requires_grad=True)
    v = torch.randn(shape, dtype=dtype, device=device, requires_grad=True)
    return q, k, v


def relative_error(a, b):
    """Elementwise relative error, safe against division by ~0."""
    num = (a - b).abs()
    den = b.abs().clamp_min(1e-6)
    return (num / den)


def check_one_case(impl_fn, batch, heads, seq_len, head_dim, causal, dtype, device, seed,
                    fwd_atol, fwd_rtol, grad_atol, grad_rtol, verbose=True):
    sm_scale = head_dim ** -0.5

    # --- custom kernel path ---
    q1, k1, v1 = make_inputs(batch, heads, seq_len, head_dim, dtype, device, seed)
    out1 = impl_fn(q1, k1, v1, causal, sm_scale)
    grad_out = torch.randn_like(out1)  # same upstream grad for both paths
    out1.backward(grad_out)

    # --- reference path (fresh identical inputs) ---
    q2, k2, v2 = make_inputs(batch, heads, seq_len, head_dim, dtype, device, seed)
    out2 = reference_attention(q2, k2, v2, causal, sm_scale)
    out2.backward(grad_out)

    results = {}

    def compare(name, a, b, atol, rtol):
        max_abs = (a - b).abs().max().item()
        rel = relative_error(a, b)
        max_rel = rel.max().item()
        mean_rel = rel.mean().item()
        passed = torch.allclose(a, b, atol=atol, rtol=rtol)
        results[name] = dict(max_abs=max_abs, max_rel=max_rel, mean_rel=mean_rel, passed=passed)
        return passed

    ok_fwd = compare("forward", out1, out2, fwd_atol, fwd_rtol)
    ok_dq = compare("dQ", q1.grad, q2.grad, grad_atol, grad_rtol)
    ok_dk = compare("dK", k1.grad, k2.grad, grad_atol, grad_rtol)
    ok_dv = compare("dV", v1.grad, v2.grad, grad_atol, grad_rtol)

    all_ok = ok_fwd and ok_dq and ok_dk and ok_dv

    if verbose:
        tag = "PASS" if all_ok else "FAIL"
        print(f"[{tag}] b={batch} h={heads} seq={seq_len} d={head_dim} "
              f"causal={causal} dtype={dtype}")
        for name, r in results.items():
            mark = "ok " if r["passed"] else "BAD"
            print(f"        {mark} {name:8s} max_abs={r['max_abs']:.3e} "
                  f"max_rel={r['max_rel']:.3e} mean_rel={r['mean_rel']:.3e}")

    return all_ok, results


def run_suite(impl_fn, device):
    """
    Shape suite chosen to stress the parts of a tiled kernel that actually
    break in practice:
      - seq_len not a multiple of the block size (boundary/padding bugs)
      - seq_len smaller than one block (degenerate tiling)
      - seq_len larger than several blocks (multi-tile accumulation)
      - head_dim values you'll actually use (Qwen2.5-1.5B uses 128 or 64
        depending on config -- check yours; both included here)
      - causal AND non-causal (masking logic is a common bug source)
      - small batch*heads (cheap, fast iteration) and one bigger case
    """
    cases = []
    for head_dim in (64, 128):
        for seq_len in (1, 17, 65, 128, 129, 383, 512, 1024):
            for causal in (True, False):
                cases.append(dict(batch=2, heads=4, seq_len=seq_len,
                                   head_dim=head_dim, causal=causal))

    all_pass = True
    failures = []

    print("=" * 70)
    print("FP32 pass (isolates algorithm correctness from bf16 precision)")
    print("=" * 70)
    for c in cases:
        ok, _ = check_one_case(
            impl_fn, dtype=torch.float32, device=device, seed=0,
            fwd_atol=1e-4, fwd_rtol=1e-4,
            grad_atol=1e-3, grad_rtol=1e-3,
            **c,
        )
        all_pass &= ok
        if not ok:
            failures.append(("fp32", c))

    print()
    print("=" * 70)
    print("BF16 pass (matches your actual training dtype)")
    print("=" * 70)
    for c in cases:
        ok, _ = check_one_case(
            impl_fn, dtype=torch.bfloat16, device=device, seed=0,
            # bf16 has ~3 decimal digits; these tolerances are deliberately
            # loose relative to fp32 -- don't tighten them without reason,
            # you'll just be chasing bf16 rounding noise, not real bugs.
            fwd_atol=3e-2, fwd_rtol=3e-2,
            grad_atol=5e-2, grad_rtol=5e-2,
            **c,
        )
        all_pass &= ok
        if not ok:
            failures.append(("bf16", c))

    print()
    print("=" * 70)
    if all_pass:
        print("ALL CASES PASSED. Kernel matches SDPA reference within tolerance.")
    else:
        print(f"{len(failures)} CASE(S) FAILED:")
        for dtype_name, c in failures:
            print(f"  - {dtype_name}: {c}")
        print()
        print("Do not use this kernel for real training until every case passes.")
        print("A failure isolated to ONE dtype (bf16 only) points at numerical")
        print("precision / rescaling order, not algorithmic logic.")
        print("A failure at specific seq_len values (e.g. non-multiples of your")
        print("block size) points at tiling/boundary handling in the kernel.")
        print("A failure in dQ/dK/dV but not forward points at the backward pass")
        print("specifically -- the recomputation or rescaling in bwd is wrong")
        print("even though forward tiling is correct.")
    print("=" * 70)

    return all_pass


def load_impl(spec):
    """spec is 'module_path:function_name', e.g. 'my_kernel:flash_attn_fwd'"""
    if ":" not in spec:
        raise ValueError("--impl must be 'module:function', e.g. 'my_kernel:flash_attn_fwd'")
    module_name, fn_name = spec.split(":", 1)
    module = importlib.import_module(module_name)
    fn = getattr(module, fn_name)
    return fn


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--impl", type=str, required=True,
        help="module:function pointing at your kernel's forward entrypoint",
    )
    parser.add_argument(
        "--quick", action="store_true",
        help="run a single small case instead of the full suite (fast iteration while debugging)",
    )
    args = parser.parse_args()

    device = get_device()
    impl_fn = load_impl(args.impl)

    if args.quick:
        ok, _ = check_one_case(
            impl_fn, batch=1, heads=2, seq_len=128, head_dim=64,
            causal=True, dtype=torch.float32, device=device, seed=0,
            fwd_atol=1e-4, fwd_rtol=1e-4, grad_atol=1e-3, grad_rtol=1e-3,
        )
        sys.exit(0 if ok else 1)

    ok = run_suite(impl_fn, device)
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
