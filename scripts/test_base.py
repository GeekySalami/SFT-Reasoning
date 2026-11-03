import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

base_model_path = "/workspace/sft_tinyllama/models/qwen2.5-1.5b"

tokenizer = AutoTokenizer.from_pretrained(base_model_path, trust_remote_code=True)

model = AutoModelForCausalLM.from_pretrained(
    base_model_path,
    dtype=torch.float16,
    device_map="auto",
    trust_remote_code=True
)
model.eval()

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