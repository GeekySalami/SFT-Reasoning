# TinyLlama Reasoning SFT Playground

[![Python](https://img.shields.io/badge/python-3.10%2B-blue.svg)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.4%2B-ee4c2c.svg)](https://pytorch.org/)
[![Transformers](https://img.shields.io/badge/%F0%9F%A4%97%20Transformers-4.46%2B-ffcc00.svg)](https://huggingface.co/docs/transformers/)
[![TRL](https://img.shields.io/badge/%F0%9F%A4%97%20TRL-0.12%2B-ffcc00.svg)](https://huggingface.co/docs/trl/)
[![License](https://img.shields.io/badge/license-none-lightgrey.svg)](#license)

A hands-on, end-to-end learning project for **supervised reasoning fine-tuning**: teaching a small
language model to *think out loud* — to emit explicit `<think> ... </think>` chain-of-thought
traces — with **LoRA SFT** on consumer-level hardware, and then measuring whether the reasoning
actually improved benchmark performance.

> The goal of this repository is **learning, not production.** Every script was written to
> understand one piece of the reasoning-SFT pipeline: dataset construction, prompt masking,
> token-budget filtering, LoRA on a tight memory budget, checkpoint resume, and benchmark-driven
> evaluation on GSM8K.

---

## Why this project exists

Reasoning fine-tuning (the style popularized by models like QwQ and DeepSeek-R1) trains a model to
produce a structured **thinking trace** before answering. This repo reproduces that recipe at
small scale to answer, hands-on:

1. **How do you build a reasoning dataset?** — A general instruction corpus
   ([Bespoke-Stratos-17k](https://huggingface.co/datasets/bespokelabs/Bespoke-Stratos-17k)) is
   normalized to ChatML, and token lengths are measured on the **fully templated** conversation
   (special tokens included) so nothing is silently truncated at train time
   (`scripts/get_data.py`).
2. **How do you teach the think tags?** — `<think>` / `</think>` are added as *special tokens*,
   the embedding table is resized, and the model learns them as explicit control tokens.
3. **How do you train on one small GPU?** — bf16, LoRA (r=16) on all attention + MLP + embedding
   modules, gradient checkpointing, batch size 1 with gradient accumulation, SDPA attention, and a
   periodic memory-cleanup callback (`scripts/sft3.py`).
4. **Does the reasoning transfer?** — The fine-tuned model is evaluated on the **GSM8K** math
   benchmark and compared against the base model conversationally
   (`scripts/benchmark_finetune.py`, `scripts/bench_batch.py`).

Despite the repository name, the experiments focus on **Qwen 2.5 1.5B** — deliberately small enough
to train locally, large enough for the effect of reasoning fine-tuning to be visible. The training
code is model-agnostic: point `MODEL_NAME` at any compatible causal LM (a local Mistral checkpoint is
included under `models/` for experimentation).

## Headline result

Full GSM8K test evaluation of the fine-tuned model (thinking trace generated before the answer):

| Model | Dataset | GSM8K test accuracy |
| --- | --- | --- |
| Qwen2.5-1.5B + reasoning LoRA | Bespoke-Stratos-17k (filtered ≤ 4,096 tokens) | **45.49 %** (1,319 samples, 0 parse failures) |

> **Honest caveats:** generation uses sampling, so runs vary; no seeds/deterministic decoding were
> fixed for this run; and the base model's score is not included above because it refuses/ignores
> the ChatML + think-tag format. Read `results/gsm8k_results.json` for per-question outputs, and
> treat the number as a learning exercise, not a leaderboard claim.

## Pipeline

```
Bespoke-Stratos-17k ──get_data.py──▶ filtered ChatML dataset (≤ 4,096 tokens)
                                              │
Qwen2.5-1.5B ── LoRA r=16 ──▶ sft3.py ────────┤
                                              ▼
                                    adapter + checkpoints (artifacts/)
                                              │
                                              ▼
                     test_qwen_chat.py (chat) / benchmark_*.py (GSM8K)
```

---

## What is included

| Area | Entry point | Purpose |
| --- | --- | --- |
| Dataset preparation | `scripts/get_data.py` | Downloads Bespoke-Stratos-17k, applies the ChatML template, measures **total templated token length**, and keeps only examples ≤ 4,096 tokens. |
| **Recommended training** | `scripts/sft3.py` | LoRA SFT with held-out eval split, checkpoint resume, SDPA attention, TensorBoard logging, and memory-cleanup callback. |
| Interactive inference | `scripts/test_qwen_chat.py` | Local chat loop for an existing adapter. |
| Single-prompt checks | `scripts/test_base.py`, `scripts/test_qwen.py`, `scripts/test_ckpt.py` | Compare the base model vs. adapters on one prompt. |
| GSM8K evaluation | `scripts/benchmark_finetune.py`, `scripts/bench_batch.py` | Sequential and batched benchmark runners; both write a JSON result file. |
| Custom attention kernel | `scripts/kernal/` | A correctness harness (`harness.py`) and SDPA reference stub (`sdpa.py`) for a future Triton/ROCm flash-attention kernel — independent of the SFT workflow. |
| Earlier experiments | `scripts/sft2.py`, `scripts/sft_llama.py` | Historical trainer variants, kept for reference only. |

**Start with `scripts/get_data.py` and `scripts/sft3.py`.** The older scripts contain
machine-specific hard-coded paths; fix their `base_model_path` / `adapter_path` constants before
relying on them.

## Requirements

- Python **3.10+** (developed on 3.12)
- PyTorch with CUDA or ROCm — training on integrated GPUs works but is slow
- Disk space for the base model (~3 GB), the filtered dataset, and checkpoints
- A Hugging Face account/token when the model or dataset requires access

## Quick start

Clone and set up an environment (uv or plain pip):

```bash
git clone <your-repository-url>
cd sft_tinyllama

# Option A: uv (respects uv.lock)
uv sync

# Option B: pip
python -m venv .venv && source .venv/bin/activate   # Windows: .venv\Scripts\activate
python -m pip install --upgrade pip
python -m pip install -e .
```

If the generic PyTorch wheel does not pick up your GPU, install the correct CUDA/ROCm build using
the [official selector](https://pytorch.org/get-started/locally/).

### 1. Prepare the reasoning dataset

```bash
python scripts/get_data.py
```

Writes the filtered dataset to `data/bespoke_stratos_filtered_4096/` and prints length
statistics. The token budget is measured on the **full templated sequence** — ChatML markers plus
`<think>` / `</think>` tokens included — so training never hits a hidden truncation cliff.

### 2. Start (or resume) training

```bash
python scripts/sft3.py
```

Adapter and checkpoints are written to `artifacts/qwen2.5-1.5b-stratos-lora/`. Re-running the
command automatically resumes from the latest checkpoint found there. Key training facts:

| Setting | Value | Why |
| --- | --- | --- |
| LoRA targets | `q/k/v/o/gate/up/down_proj` + `embed_tokens` + `lm_head` | New think tokens must be trainable, so LoRA goes *on* the embeddings instead of full-module saving |
| Effective batch size | 1 × 32 gradient accumulation | Fits tiny VRAM |
| Precision | bf16 + gradient checkpointing | Memory |
| Eval | 450 held-out examples, `eval_loss` every 100 steps | Watch for overfitting |
| Resume | automatic via checkpoint detection | Crash-friendly |

### 3. Monitor

```bash
tensorboard --logdir artifacts/qwen2.5-1.5b-stratos-lora
```

### 4. Talk to your model

Point `adapter_path` in `scripts/test_qwen_chat.py` at your output directory, then:

```bash
python scripts/test_qwen_chat.py
```

### 5. Benchmark on GSM8K

```bash
python scripts/benchmark_finetune.py   # lower memory, one prompt at a time
python scripts/bench_batch.py          # faster, tune BATCH_SIZE to available VRAM
```

Both download the GSM8K test split on first use and write a JSON result file (question,
ground truth, prediction, per-item correctness). Generation is sampled — fix seeds and use
deterministic decoding before quoting numbers.

## Configuration overrides

The maintained scripts read these environment variables, so weights/data can live outside the repo:

```bash
MODEL_NAME=/data/Qwen2.5-1.5B \
DATA_DIR=/data/stratos-4096 \
OUTPUT_DIR=/experiments/qwen-lora \
python scripts/sft3.py
```

| Variable | Used by | Default |
| --- | --- | --- |
| `MODEL_NAME` | `get_data.py`, `sft3.py` | `Qwen/Qwen2.5-1.5B` |
| `DATA_DIR` | `get_data.py`, `sft3.py` | `data/bespoke_stratos_filtered_4096` |
| `OUTPUT_DIR` | `sft3.py` | `artifacts/qwen2.5-1.5b-stratos-lora` |

## Using an adapter on your own

LoRA adapters do **not** include the base model. Load the same base checkpoint, restore the
tokenizer, re-add the think tokens, resize the embeddings, then attach the adapter with PEFT — the
exact recipe is in `scripts/test_qwen.py`:

```python
tokenizer = AutoTokenizer.from_pretrained(base_model_path)
tokenizer.add_special_tokens({"additional_special_tokens": ["<think>", "</think>"]})
model = AutoModelForCausalLM.from_pretrained(base_model_path, dtype=torch.float16)
model.resize_token_embeddings(len(tokenizer))
model = PeftModel.from_pretrained(model, adapter_path)
```

Prompt format must match training exactly:

```
<|im_start|>user
{question}<|im_end|>
<|im_start|>assistant
```

## Repository layout

```text
.
├── data/                        # generated datasets (gitignored; build with get_data.py)
├── models/                      # downloaded base weights (gitignored)
├── artifacts/                   # training products — adapters, checkpoints (created locally)
├── results/                     # benchmark output (gsm8k_results.json committed)
├── scripts/
│   ├── get_data.py              # corpus preparation + token-budget filtering
│   ├── sft3.py                  # recommended trainer (start here)
│   ├── sft2.py, sft_llama.py    # earlier trainer variants, for reference
│   ├── test_base.py             # base-model smoke test
│   ├── test_qwen.py             # + adapter smoke test
│   ├── test_ckpt.py             # + checkpoint smoke test
│   ├── test_qwen_chat.py        # interactive chat
│   ├── benchmark_finetune.py    # GSM8K, sequential (low memory)
│   ├── bench_batch.py           # GSM8K, batched (fast)
│   └── kernal/                  # Triton/ROCm flash-attention harness (independent)
├── pyproject.toml               # dependencies + project metadata
└── README.md
```

## Troubleshooting

| Symptom | Likely fix |
| --- | --- |
| Out-of-memory error | Lower `MAX_SEQ_LEN`, keep `BATCH_SIZE=1` and raise gradient accumulation, or use a larger GPU. |
| Model download / access error | Run `huggingface-cli login`, or point `MODEL_NAME` at a local weights directory. |
| `DATA_DIR` not found | Run `python scripts/get_data.py` first, or set `DATA_DIR`. |
| Tokenizer / embedding size mismatch | Re-add `<think>` / `</think>` and call `resize_token_embeddings` before loading the adapter (see `test_qwen.py`). |
| Hard-coded path errors in older scripts | Edit `base_model_path` / `adapter_path` constants in `test_*.py` and `benchmark_*.py`. |
| ROCm-specific error | Install a ROCm-compatible PyTorch build. Kernel work in `scripts/kernal/` is optional and not needed for SFT. |

## Lessons that shaped this repo

- **Measure token length after templating.** Filtering by raw character length silently trains on
  truncated reasoning traces — the single biggest correctness trap in this pipeline.
- **Think tokens are part of the model, not the data.** Adding them as special tokens, resizing
  embeddings, and exercising the new rows via LoRA is what makes `<think>` / `</think>` generation reliable.
- **Memory is a design constraint, not an afterthought.** bf16 + SDPA + gradient checkpointing +
  fused AdamW + periodic cache clearing is what lets a 1.5B model train LoRA on an integrated GPU.
- **Evaluate, don't eyeball.** None of this felt "real" until GSM8K produced a JSON with a rate —
  which is also why per-question outputs ship alongside the aggregate number.

## License & responsible use

No license file yet — add one before relying on or redistributing this code. This repository
contains **code, configuration, and one GSM8K result file**: model weights, datasets, checkpoints,
and logs are deliberately excluded from Git. Before training or publishing derived artifacts, respect the
licenses of Qwen, Bespoke-Stratos, GSM8K, and every dependency. Reasoning traces can contain
unwanted content — review outputs before deploying anything. If you publish an adapter, include
the base-model revision, dataset revision, training configuration, hardware, and evaluation method
so others can reproduce it.
