"""
Build a local, length-filtered copy of the training corpus.

Key correctness point: token budget must be measured on the FULL
tokenized sequence -- prompt template tokens (<|im_start|>, <|im_end|>,
role markers) + the <think>/</think> tokens + content -- not just the
raw text length. A conversation that looks short in characters can
still blow past 4096 once special tokens and template wrapping are
counted, and the reverse is also true.

This script:
  1. Loads the same dataset/tokenizer your training script uses
  2. Applies the SAME prompt/completion template as tokenize_and_mask
  3. Measures total token length AFTER special tokens are added
  4. Keeps only examples where total length <= MAX_SEQ_LEN
  5. Saves the filtered set locally so training loads it directly,
     with no silent truncation surprises at train time
"""

import os
from pathlib import Path
from datasets import load_dataset
from transformers import AutoTokenizer

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
# A Hugging Face model ID works out of the box. Set MODEL_NAME to a local path
# when weights have already been downloaded.
MODEL_NAME = os.environ.get("MODEL_NAME", "Qwen/Qwen2.5-1.5B")
MAX_SEQ_LEN = 4096
OUTPUT_DIR = Path(
    os.environ.get("DATA_DIR", REPOSITORY_ROOT / "data" / "bespoke_stratos_filtered_4096")
)

# Set to True to strip <think>...</think> spans from the assistant
# message before tokenizing/measuring length. See discussion below --
# this changes WHAT the model is trained on, not just data hygiene.
STRIP_THINK_TAGS = False


def strip_think_spans(text: str) -> str:
    """Remove <think>...</think> blocks (including tags) from text."""
    import re
    return re.sub(r"<think>.*?</think>\s*", "", text, flags=re.DOTALL)


def main():
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, use_fast=True)

    # Must match training script exactly -- if you add these tokens at
    # train time, you need them here too, since they change token counts.
    special_tokens = ["<think>", "</think>"]
    tokenizer.add_special_tokens({"additional_special_tokens": special_tokens})

    dataset = load_dataset("bespokelabs/Bespoke-Stratos-17k", split="train")
    dataset = dataset.shuffle(seed=42)  # keep same seed as training for reproducibility

    def measure_and_filter(example):
        conversations = example.get("conversations", [])
        user_msg = next((t["value"] for t in conversations if t["from"] == "user"), "")
        assistant_msg = next((t["value"] for t in conversations if t["from"] == "assistant"), "")

        assistant_msg = assistant_msg.replace("<|begin_of_thought|>", "<think>").replace(
            "<|end_of_thought|>", "</think>"
        )

        if STRIP_THINK_TAGS:
            assistant_msg = strip_think_spans(assistant_msg)

        # Same template as tokenize_and_mask -- special tokens
        # (<|im_start|>, <|im_end|>, <think>/</think>) are part of the
        # actual token stream, so they must be included when measuring
        # length, not just the raw text.
        prompt_ids = tokenizer.encode(
            f"<|im_start|>user\n{user_msg}<|im_end|>\n<|im_start|>assistant\n",
            add_special_tokens=False,
        )
        completion_ids = tokenizer.encode(f"{assistant_msg}<|im_end|>", add_special_tokens=False)

        total_len = len(prompt_ids) + len(completion_ids)

        return {
            "user": user_msg,
            "assistant": assistant_msg,
            "total_tokens": total_len,
            "keep": total_len <= MAX_SEQ_LEN,
        }

    processed = dataset.map(
        measure_and_filter,
        remove_columns=dataset.column_names,
        num_proc=4,
        desc="Measuring token lengths",
    )

    before = len(processed)
    filtered = processed.filter(lambda ex: ex["keep"], desc="Filtering to <= MAX_SEQ_LEN")
    after = len(filtered)

    # Drop the helper columns, keep clean data for training to consume
    filtered = filtered.remove_columns(["keep"])

    OUTPUT_DIR.parent.mkdir(parents=True, exist_ok=True)
    filtered.save_to_disk(OUTPUT_DIR)

    lengths = processed["total_tokens"]
    print(f"Total examples:    {before}")
    print(f"Kept (<= {MAX_SEQ_LEN}): {after} ({100 * after / before:.1f}%)")
    print(f"Dropped (> {MAX_SEQ_LEN}): {before - after}")
    print(f"Min length:  {min(lengths)}")
    print(f"Max length:  {max(lengths)}")
    print(f"Mean length: {sum(lengths) / len(lengths):.1f}")
    print(f"Saved to: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
