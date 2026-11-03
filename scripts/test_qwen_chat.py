import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

base_model_path = "/workspace/sft_tinyllama/models/qwen2.5-1.5b"
adapter_path    = "/workspace/sft_tinyllama/scripts/outputs/qwen2.5-1.5b_integrated_gpu/"

tokenizer = AutoTokenizer.from_pretrained(base_model_path, trust_remote_code=True)
tokenizer.add_special_tokens(
    {"additional_special_tokens": ["<think>", "</think>"]}
)
print(f"Tokenizer size: {len(tokenizer)}")

print("Loading base model...")
model = AutoModelForCausalLM.from_pretrained(
    base_model_path,
    dtype=torch.float16,
    device_map="auto",
    trust_remote_code=True
)
model.resize_token_embeddings(len(tokenizer))
model.tie_weights()

print("Loading adapter...")
model = PeftModel.from_pretrained(model, adapter_path, is_trainable=False)
model.eval()

# Determine actual device
device = next(model.parameters()).device
print(f"Model device: {device}")
print(f"Embed device: {model.base_model.model.model.embed_tokens.weight.device}")

print("\nModel loaded. Entering chat loop. Type 'exit' or 'quit' to stop.")

prompt_history = ""

while True:
    try:
        user_input = input("\nYou: ")
        if user_input.strip().lower() in ['exit', 'quit']:
            break
        if not user_input.strip():
            continue

        # Fresh prompt each turn, no history
        prompt = f"<|im_start|>user\n{user_input}<|im_end|>\n<|im_start|>assistant\n"

        inputs = tokenizer(prompt, return_tensors="pt").to(device)
        print(f"[debug] inputs device: {inputs.input_ids.device}, tokens: {inputs.input_ids.shape[1]}")

        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=2048,
                do_sample=True,
                temperature=0.7,
                repetition_penalty=1.1,
                pad_token_id=tokenizer.eos_token_id
            )

        input_length = inputs.input_ids.shape[1]
        generated_tokens = outputs[0][input_length:]
        response_clean = tokenizer.decode(generated_tokens, skip_special_tokens=False).strip()
        print(f"\nA: {response_clean}")

    except KeyboardInterrupt:
        print("\nExiting.")
        break
    except EOFError:
        break