import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

base_model_path = "/workspace/sft_tinyllama/models/qwen2.5-1.5b"
adapter_path    = "/workspace/sft_tinyllama/scripts/outputs/qwen2.5-1.5b_integrated_gpu/"

# Load tokenizer from base model (151665) then force to trained size
tokenizer = AutoTokenizer.from_pretrained(base_model_path, trust_remote_code=True)

# The adapter was trained with vocab 151667 (2 extra tokens added during training)
# Re-add them so tokenizer matches the checkpoint
tokenizer.add_special_tokens(
    {"additional_special_tokens": ["<think>", "</think>"]}
)
print(f"Tokenizer size: {len(tokenizer)}")  # should print 151667

# Load base model
model = AutoModelForCausalLM.from_pretrained(
    base_model_path,
    dtype=torch.float16,
    device_map="auto",
    trust_remote_code=True
)

# Resize to match trained checkpoint
model.resize_token_embeddings(len(tokenizer))  # 151667
model.tie_weights()
print(f"Embed size: {model.model.embed_tokens.weight.shape[0]}")  # should print 151667

# Load adapter
model = PeftModel.from_pretrained(
    model,
    adapter_path,
    is_trainable=False
)
model.eval()

# Inference
prompt = "<|im_start|>user\nWhat are the main causes of World War 1?<|im_end|>\n<|im_start|>assistant\n"
inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

with torch.no_grad():
    outputs = model.generate(
        **inputs,
        max_new_tokens=1024,
        do_sample=True,
        temperature=0.7,
        repetition_penalty=1.1,
        pad_token_id=tokenizer.eos_token_id
    )

print(tokenizer.decode(outputs[0], skip_special_tokens=False))