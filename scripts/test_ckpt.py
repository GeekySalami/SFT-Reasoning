import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

# 1. Define paths
base_model_path = "/workspace/sft_tinyllama/models/qwen2.5-1.5b"
adapter_path = "/workspace/sft_tinyllama/scripts/outputs/qwen2.5-1.5b_integrated_gpu/" # Updated to match your training output dir

# 2. Load Tokenizer from ADAPTER PATH (Crucial!)
# This ensures it loads the tokenizer with the <think> tokens included (size 151667).
tokenizer = AutoTokenizer.from_pretrained(adapter_path, trust_remote_code=True)

# 3. Load Base Model normally
model = AutoModelForCausalLM.from_pretrained(
    base_model_path,
    torch_dtype=torch.float16,
    device_map="auto",
    trust_remote_code=True
)

# 4. Resize embeddings to match the new tokenizer (151667)
model.resize_token_embeddings(len(tokenizer))

# 5. Load the PEFT adapter
model = PeftModel.from_pretrained(model, adapter_path)

# 6. Prepare Input using your specific ChatML training format
prompt = "<|im_start|>user\nWhat are the main causes of World War 1?<|im_end|>\n<|im_start|>assistant\n"
inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

# 7. Generate
outputs = model.generate(
    **inputs, 
    max_new_tokens=400, # Increased slightly to allow room for <think> reasoning
    do_sample=True,
    temperature=0.7,
    repetition_penalty=1.1,
    pad_token_id=tokenizer.eos_token_id
)

# 8. Decode (We set skip_special_tokens=False here so you can actually see the <think> tags)
print(tokenizer.decode(outputs[0], skip_special_tokens=False))