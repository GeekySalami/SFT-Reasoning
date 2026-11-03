import os
import torch
# Force aggressive memory clearing
os.environ["PYTORCH_HIP_ALLOC_CONF"] = "expandable_segments:True"

from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer, DataCollatorForSeq2Seq
from peft import LoraConfig, get_peft_model
from trl import SFTTrainer, SFTConfig
from transformers import TrainerCallback
import torch
import gc

class MemoryCleanupCallback(TrainerCallback):
    def on_step_end(self, args, state, control, **kwargs):
        if state.global_step % 50 == 0:
            gc.collect()
            torch.cuda.empty_cache()

# Global Settings
USE_BF16 = True 
MAX_SEQ_LEN = 4096  # Compromise length for reasoning vs memory
MODEL_NAME = "/workspace/sft_tinyllama/models/qwen2.5-1.5b"
OUTPUT_DIR = "outputs/qwen2.5-1.5b_integrated_gpu"

def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, use_fast=True)
    
    # ── Model Loading (Extreme Memory Mode) ──────────────────────────────────
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        torch_dtype=torch.bfloat16,
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
    dataset = load_dataset("bespokelabs/Bespoke-Stratos-17k", split="train")
    dataset = dataset.shuffle(seed=42).select(range(5000))

    def tokenize_and_mask(example):
        conversations = example.get("conversations", [])
        user_msg = next((t["value"] for t in conversations if t["from"] == "user"), "")
        assistant_msg = next((t["value"] for t in conversations if t["from"] == "assistant"), "")
        
        # Standardize thinking tags
        assistant_msg = assistant_msg.replace("<|begin_of_thought|>", "<think>").replace("<|end_of_thought|>", "</think>")
        
        prompt_ids = tokenizer.encode(f"<|im_start|>user\n{user_msg}<|im_end|>\n<|im_start|>assistant\n", add_special_tokens=False)
        completion_ids = tokenizer.encode(f"{assistant_msg}<|im_end|>", add_special_tokens=False)
        
        input_ids = (prompt_ids + completion_ids) #[:MAX_SEQ_LEN]
        labels = ([-100] * len(prompt_ids) + completion_ids) #[:MAX_SEQ_LEN]
        
        return {"input_ids": input_ids, "labels": labels, "attention_mask": [1] * len(input_ids)}




########################################################################################################
    dataset = dataset.map(tokenize_and_mask, remove_columns=dataset.column_names, num_proc=4)

    # ── Training Arguments ───────────────────────────────────────────────────
    training_args = SFTConfig(
        output_dir=OUTPUT_DIR,
        # max_seq_length=MAX_SEQ_LEN,
        per_device_train_batch_size=1, # Mandatory for integrated GPUs
        gradient_accumulation_steps=32, # Effective batch size 32
        
        learning_rate=1e-4,
        lr_scheduler_type="cosine",
        warmup_ratio=0.1,
        
        bf16=True,
        # Use the most basic optimizer to save memory
        optim="adamw_torch_fused", 
        gradient_checkpointing=True,
        
        logging_steps=50,
        save_strategy="steps",
        save_steps=100,
        save_total_limit=2,
        report_to="tensorboard",
        logging_dir=f"{OUTPUT_DIR}/logs",
    )

    trainer = SFTTrainer(
        model=model,
        args=training_args,
        train_dataset=dataset,
        data_collator=DataCollatorForSeq2Seq(tokenizer, padding=True, return_tensors="pt"),
        callbacks=[MemoryCleanupCallback()]
    )

    trainer.train()
    trainer.save_model(OUTPUT_DIR)

if __name__ == "__main__":
    main()