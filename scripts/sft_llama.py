import os
import torch

os.environ["TENSORBOARD_LOGGING_DIR"] = "./logs"

from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import LoraConfig
from trl import SFTTrainer, SFTConfig


MODEL_NAME = "/workspace/sft_tinyllama/models/qwen2.5-1.5b"
OUTPUT_DIR = "outputs/qwen2.5-1.5b_think"


def main():

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # ── Tokenizer ──────────────────────────────────────────────────────────────
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, use_fast=True)

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token


    # Add thinking tokens so the model learns <think> / </think> as special units
    # tokenizer.add_special_tokens(
    #     {"additional_special_tokens": ["<think>", "</think>"]}
    # )
    

    # ── Model ──────────────────────────────────────────────────────────────────
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32,
        low_cpu_mem_usage=True,

    )

    # model.resize_token_embeddings(len(tokenizer))

    # # Initialize new token embeddings from mean of existing embeddings
    # with torch.no_grad():
    #     existing = model.model.embed_tokens.weight[:-2]   # all except new 2
    #     mean_embedding = existing.mean(dim=0)
    #     model.model.embed_tokens.weight[-2] = mean_embedding  # <think>
    #     model.model.embed_tokens.weight[-1] = mean_embedding  # </think>

    # Check first
    think_id = tokenizer.convert_tokens_to_ids("<think>")
    print(f"<think> token id: {think_id}")

    # Only add if truly missing
    if think_id == tokenizer.unk_token_id:
        tokenizer.add_special_tokens(
            {"additional_special_tokens": ["<think>", "</think>"]}
        )
        model.resize_token_embeddings(len(tokenizer))
        with torch.no_grad():
            existing = model.model.embed_tokens.weight[:-2]
            mean_embedding = existing.mean(dim=0)
            model.model.embed_tokens.weight[-2] = mean_embedding
            model.model.embed_tokens.weight[-1] = mean_embedding
    else:
        print("Qwen already has <think> tokens — no resize needed")


    # ── Dataset ────────────────────────────────────────────────────────────────
    # Use ALL 17k examples — too few examples caused memorization before
    dataset = load_dataset("bespokelabs/Bespoke-Stratos-17k", split="train")

    print(f"Dataset size: {len(dataset)}")
    print(f"Columns: {dataset.column_names}")

    # Filter out very long examples to stay within 2048 tokens
    # dataset = dataset.filter(
    #     lambda x: (
    #         len(x.get("instruction", "").split()) +
    #         len(x.get("reasoning", x.get("output", "")).split()) < 1500
    #     )
    # )

    print(f"After filtering: {len(dataset)} examples")

    # ── Formatting ─────────────────────────────────────────────────────────────
    # def format_example(example):
    #     instruction = example.get("instruction", example.get("input", ""))
    #     reasoning   = example.get("reasoning", "")
    #     answer      = example.get("output", "")

    #     if reasoning:
    #         assistant_turn = f"<think>\n{reasoning}\n</think>\n{answer}"
    #     else:
    #         assistant_turn = answer

    #     text = (
    #         f"### User:\n{instruction}\n\n"
    #         f"### Assistant:\n{assistant_turn}"
    #     )
    #     return {"text": text}

    def format_example(example):
        instruction = example.get("instruction", example.get("input", ""))
        reasoning   = example.get("reasoning", "")
        answer      = example.get("output", "")

        if reasoning:
            assistant_turn = f"<think>\n{reasoning}\n</think>\n{answer}"
        else:
            assistant_turn = answer

        text = (
            f"<|im_start|>user\n{instruction}<|im_end|>\n"
            f"<|im_start|>assistant\n{assistant_turn}<|im_end|>"
        )
        return {"text": text}


    dataset = dataset.map(
        format_example,
        remove_columns=dataset.column_names,
        num_proc=4,
    )

    # ── LoRA ───────────────────────────────────────────────────────────────────
    peft_config = LoraConfig(
        r=8,                # was 16 — lower rank prevents memorization
        lora_alpha=16,      # was 32
        lora_dropout=0.1,   # was 0.05 — higher dropout fights overfitting
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=["q_proj", "v_proj"],  # fewer modules, less overfitting
        modules_to_save=["embed_tokens", "lm_head"],
    )

    # ── Training config ────────────────────────────────────────────────────────
    training_args = SFTConfig(
        output_dir=OUTPUT_DIR,
        dataset_text_field="text",

        num_train_epochs=1,     # start with 1 — 1 epoch is enough to learn the pattern
        max_steps=-1,

        per_device_train_batch_size=4,
        gradient_accumulation_steps=16,  # effective batch = 16

        learning_rate=1e-4,     # was 4e-4 — lower LR prevents memorization
        max_grad_norm=0.3,      # was 1.0 — tighter clipping
        warmup_steps=200,       # longer warmup for stability
        lr_scheduler_type="cosine",

        logging_steps=10,
        save_steps=200,
        save_total_limit=2,

        dataset_num_proc=8,
        report_to="tensorboard",

        bf16=torch.cuda.is_available(),
        packing=False,
        optim="adamw_torch",
    )
    # Force the unfrozen embedding and lm_head layers to float32
    for name, param in model.named_parameters():
        if "embed_tokens" in name or "lm_head" in name:
            param.data = param.data.to(torch.float32)
    # ── Trainer ────────────────────────────────────────────────────────────────
    trainer = SFTTrainer(
        model=model,
        args=training_args,
        train_dataset=dataset,
        peft_config=peft_config,
        processing_class=tokenizer,
    )

    trainer.train()

    trainer.save_model(OUTPUT_DIR)
    tokenizer.save_pretrained(OUTPUT_DIR)
    print(f"\nModel saved to {OUTPUT_DIR}")


if __name__ == "__main__":
    main()