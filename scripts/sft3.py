import os
from pathlib import Path
os.environ["TORCH_ROCM_AOTRITON_ENABLE_EXPERIMENTAL"] = "1"
# Force aggressive memory clearing
os.environ["PYTORCH_HIP_ALLOC_CONF"] = "expandable_segments:True"

import torch
import gc

from datasets import load_from_disk
from transformers import AutoModelForCausalLM, AutoTokenizer, DataCollatorForSeq2Seq, TrainerCallback
from transformers.trainer_utils import get_last_checkpoint
from peft import LoraConfig, get_peft_model
from trl import SFTTrainer, SFTConfig


class MemoryCleanupCallback(TrainerCallback):
    def on_step_end(self, args, state, control, **kwargs):
        if state.global_step % 50 == 0:
            gc.collect()
            torch.cuda.empty_cache()


# Global Settings
USE_BF16 = True
MAX_SEQ_LEN = 4096  # Compromise length for reasoning vs memory
REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
# Set MODEL_NAME to a local model directory to avoid downloading from the Hub.
MODEL_NAME = os.environ.get("MODEL_NAME", "Qwen/Qwen2.5-1.5B")
DATA_DIR = Path(
    os.environ.get("DATA_DIR", REPOSITORY_ROOT / "data" / "bespoke_stratos_filtered_4096")
)
OUTPUT_DIR = Path(
    os.environ.get(
        "OUTPUT_DIR", REPOSITORY_ROOT / "artifacts" / "qwen2.5-1.5b-stratos-lora"
    )
)


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # ── Detect existing checkpoint up front ──────────────────────────────────
    # (Needed later for trainer.train(), but useful to know now for logging.)
    last_checkpoint = get_last_checkpoint(OUTPUT_DIR) if os.path.isdir(OUTPUT_DIR) else None
    if last_checkpoint is not None:
        print(f"[resume] Found existing checkpoint, will resume from: {last_checkpoint}")
    else:
        print("[resume] No checkpoint found, starting fresh training run.")

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, use_fast=True)

    # ── Model Loading (Extreme Memory Mode) ──────────────────────────────────
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        dtype=torch.bfloat16,
        low_cpu_mem_usage=True,
        # Use 'eager' if SDPA continues to OOM during the loss step
        attn_implementation="sdpa",
    )

    # Add tokens and resize
    special_tokens = ["<think>", "</think>"]
    tokenizer.add_special_tokens({"additional_special_tokens": special_tokens})
    model.resize_token_embeddings(len(tokenizer))

    # ── LoRA Configuration (The Memory-Saving Secret) ────────────────────────
    # We remove modules_to_save and put LoRA adapters ON the embeddings instead
    peft_config = LoraConfig(
        r=16,
        lora_alpha=32,
        target_modules=["q_proj", "v_proj", "k_proj", "o_proj", "gate_proj", "up_proj", "down_proj", "embed_tokens", "lm_head"],
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM",
    )

    # Manually wrap model to ensure gradients are handled correctly for embeddings
    model = get_peft_model(model, peft_config)
    model.gradient_checkpointing_enable()

    # ── Dataset Processing ───────────────────────────────────────────────────
    dataset = load_from_disk(str(DATA_DIR))
    dataset = dataset.shuffle(seed=42)
    eval_set = dataset.select(range(450))
    train_dataset = dataset.select(range(450, len(dataset)))

    def tokenize_and_mask(example):
        user_msg = example["user"]
        assistant_msg = example["assistant"]
        # <think> tag replacement already done during corpus building, skip here

        prompt_ids = tokenizer.encode(f"<|im_start|>user\n{user_msg}<|im_end|>\n<|im_start|>assistant\n", add_special_tokens=False)
        completion_ids = tokenizer.encode(f"{assistant_msg}<|im_end|>", add_special_tokens=False)

        input_ids = prompt_ids + completion_ids
        labels = [-100] * len(prompt_ids) + completion_ids

        return {"input_ids": input_ids, "labels": labels, "attention_mask": [1] * len(input_ids)}

    train_dataset = train_dataset.map(tokenize_and_mask, remove_columns=dataset.column_names, num_proc=4)
    eval_set = eval_set.map(tokenize_and_mask, remove_columns=eval_set.column_names, num_proc=4)

    # ── Training Arguments ───────────────────────────────────────────────────
    training_args = SFTConfig(
        output_dir=OUTPUT_DIR,
        max_length=MAX_SEQ_LEN,
        per_device_train_batch_size=1,  # Mandatory for integrated GPUs
        gradient_accumulation_steps=32,  # Effective batch size 32

        learning_rate=1e-4,
        lr_scheduler_type="cosine",
        warmup_steps=90,

        bf16=True,
        # Use the most basic optimizer to save memory
        optim="adamw_torch_fused",
        gradient_checkpointing=True,

        logging_steps=50,
        save_strategy="steps",
        save_steps=100,
        save_total_limit=2,
        report_to="tensorboard",
        loss_type="nll",
        # logging_dir=f"{OUTPUT_DIR}/logs",

        eval_strategy="steps",
        eval_steps=100,
        metric_for_best_model="eval_loss",
        per_device_eval_batch_size=1,
        load_best_model_at_end=True,
    )

    trainer = SFTTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_set,
        data_collator=DataCollatorForSeq2Seq(tokenizer, padding=True, return_tensors="pt"),
        callbacks=[MemoryCleanupCallback()],
    )

    # ── Train (auto-resumes if a checkpoint was found) ───────────────────────
    trainer.train(resume_from_checkpoint=last_checkpoint)
    trainer.save_model(OUTPUT_DIR)


if __name__ == "__main__":
    main()
