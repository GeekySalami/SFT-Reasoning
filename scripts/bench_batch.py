import torch
import json
import re
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel
from datasets import load_dataset
from tqdm import tqdm
from datetime import datetime

# Config
base_model_path = "/workspace/sft_tinyllama/models/qwen2.5-1.5b"
adapter_path    = "/workspace/sft_tinyllama/scripts/outputs/qwen2.5-1.5b_integrated_gpu/"
OUTPUT_PATH     = "/workspace/sft_tinyllama/gsm8k_results.json"
MAX_NEW_TOKENS  = 2048
BATCH_SIZE      = 16   # tune to fit VRAM; try 4, 8, 16

# Load tokenizer
print("Loading tokenizer...")
tokenizer = AutoTokenizer.from_pretrained(base_model_path, trust_remote_code=True)
tokenizer.add_special_tokens({"additional_special_tokens": ["<think>", "</think>"]})
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token
# Decoder-only models must left-pad so every sequence ends at the same position
tokenizer.padding_side = "left"
print(f"Vocab size: {len(tokenizer)}")

# Load model
print("Loading base model...")
model = AutoModelForCausalLM.from_pretrained(
    base_model_path,
    torch_dtype=torch.float16,
    device_map="auto",
    trust_remote_code=True,
)
model.resize_token_embeddings(len(tokenizer))
model.tie_weights()

print("Loading adapter...")
model = PeftModel.from_pretrained(model, adapter_path, is_trainable=False)
model.eval()

device = next(model.parameters()).device
print(f"Model on: {device}\n")

# Dataset
print("Loading GSM8K test split...")
dataset = load_dataset("gsm8k", "main", split="test")
samples = list(dataset)
print(f"Total samples: {len(samples)}\n")

# Prompt builder
def make_prompt(question: str) -> str:
    system = (
        "You are a math assistant. "
        "Think step by step inside <think></think>, "
        "then give your final answer inside <|begin_of_solution|><|end_of_solution|> "
        "ending with #### <number>."
    )
    try:
        return tokenizer.apply_chat_template(
            [
                {"role": "system", "content": system},
                {"role": "user",   "content": question},
            ],
            tokenize=False,
            add_generation_prompt=True,
        )
    except Exception:
        return (
            f"<|im_start|>system\n{system}<|im_end|>\n"
            f"<|im_start|>user\n{question}<|im_end|>\n"
            f"<|im_start|>assistant\n"
        )

# Answer extraction
def extract_solution_block(text: str) -> str:
    match = re.search(r"<\|begin_of_solution\|>(.*?)<\|end_of_solution\|>", text, re.DOTALL)
    return match.group(1).strip() if match else text

def strip_think(text: str) -> str:
    return re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL).strip()

def extract_answer(text: str):
    clean = strip_think(extract_solution_block(text))

    m = re.search(r"####\s*([-\d,\.]+)", clean)
    if m:
        return m.group(1).replace(",", "").strip()

    m = re.search(r"\\boxed\{([^}]+)\}", clean)
    if m:
        return m.group(1).replace(",", "").strip()

    m = re.search(
        r"(?:the\s+answer\s+is|therefore[,\s]+|=\s*)\$?\s*([-\d,\.]+)",
        clean, re.IGNORECASE
    )
    if m:
        return m.group(1).replace(",", "").strip()

    matches = re.findall(r"=\s*([-\d,\.]+)\s*$", clean, re.MULTILINE)
    if matches:
        return matches[-1].replace(",", "").strip()

    nums = re.findall(r"\b(\d[\d,]*)\b", clean)
    if nums:
        return nums[-1].replace(",", "").strip()

    return None

def extract_gt(answer_text: str):
    m = re.search(r"####\s*([-\d,\.]+)", answer_text)
    return m.group(1).replace(",", "").strip() if m else None

def normalize(ans):
    if ans is None:
        return None
    try:
        val = float(ans.replace(",", ""))
        return int(val) if val == int(val) else round(val, 4)
    except Exception:
        return None

# Batched inference
def run_batch(batch_samples: list) -> list:
    """Run a batch of samples through the model, return list of response strings."""
    prompts = [make_prompt(s["question"]) for s in batch_samples]

    inputs = tokenizer(
        prompts,
        return_tensors="pt",
        truncation=True,
        max_length=1024,
        padding=True,          # left-pad to longest sequence in batch
    ).to(device)

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=MAX_NEW_TOKENS,
            do_sample=True,
            temperature=0.7,
            repetition_penalty=1.1,
            pad_token_id=tokenizer.eos_token_id,
        )

    responses = []
    for i, out in enumerate(outputs):
        # Each sequence was left-padded to the same length, so strip exactly
        # the number of input tokens that belonged to this sequence.
        input_len = inputs.input_ids.shape[1]   # shared padded length
        gen_tokens = out[input_len:]
        response = tokenizer.decode(gen_tokens, skip_special_tokens=False)
        response = response.replace("<|im_end|>", "").strip()
        responses.append(response)

    return responses

# Benchmark loop
correct      = 0
total        = 0
failed_parse = 0
results      = []
start_time   = datetime.now()

# Split into batches
batches = [samples[i:i + BATCH_SIZE] for i in range(0, len(samples), BATCH_SIZE)]

for batch in tqdm(batches, desc="Benchmarking"):
    responses = run_batch(batch)

    for sample, response in zip(batch, responses):
        question     = sample["question"]
        ground_truth = extract_gt(sample["answer"])

        predicted  = extract_answer(response)
        gt_val     = normalize(ground_truth)
        pred_val   = normalize(predicted)
        is_correct = gt_val is not None and pred_val is not None and gt_val == pred_val

        if predicted is None:
            failed_parse += 1
        if is_correct:
            correct += 1
        total += 1

        solution = extract_solution_block(response)
        status   = "✓" if is_correct else "✗"
        print(f"\n[{total}] {status}  GT: {ground_truth}  |  Extracted: {predicted}")
        print(f"     Solution: {solution[:200].strip()}")
        print(f"Response: {response}")

        results.append({
            "question":      question,
            "ground_truth":  ground_truth,
            "predicted":     predicted,
            "is_correct":    is_correct,
            "full_response": response,
        })

# Final stats
elapsed  = datetime.now() - start_time
accuracy = correct / total * 100

print("\n" + "="*52)
print("  GSM8K Benchmark Results")
print("="*52)
print(f"  Accuracy        : {accuracy:.2f}%  ({correct}/{total})")
print(f"  Failed to parse : {failed_parse}")
print(f"  Elapsed         : {elapsed}")
print("="*52)

wrong      = [r for r in results if not r["is_correct"]]
truncated  = [r for r in wrong if "</think>" not in r["full_response"]
                               and "<|end_of_solution|>" not in r["full_response"]]
parse_fail = [r for r in wrong if r["predicted"] is None]

print(f"\n  Wrong answers   : {len(wrong)}")
print(f"  Likely truncated: {len(truncated)}  (no closing tag found)")
print(f"  Parse failures  : {len(parse_fail)}")

if truncated:
    print("\n  -> If truncated count is high, raise MAX_NEW_TOKENS")
if parse_fail:
    print("\n  First parse failure tail:")
    print("  ..." + parse_fail[0]["full_response"][-200:])

# Save
with open(OUTPUT_PATH, "w") as f:
    json.dump({
        "accuracy":     accuracy,
        "correct":      correct,
        "total":        total,
        "failed_parse": failed_parse,
        "elapsed_sec":  elapsed.total_seconds(),
        "results":      results,
    }, f, indent=2)

print(f"\nResults saved -> {OUTPUT_PATH}")