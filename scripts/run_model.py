from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel
import torch

# Base model
base_model_path = "Qwen/Qwen2.5-72B-Instruct"

# Checkpoint path (update with the latest checkpoint)
checkpoint_path = "/mount/sdb/llama_output/checkpoint-468"

# Load tokenizer
tokenizer = AutoTokenizer.from_pretrained(base_model_path, trust_remote_code=True)

# Load base model
model = AutoModelForCausalLM.from_pretrained(base_model_path, torch_dtype=torch.bfloat16, device_map="auto")

# Load fine-tuned LoRA checkpoint
model = PeftModel.from_pretrained(model, checkpoint_path)

# Move model to GPU
model.eval()

def chat(prompt, max_length=16384):
    inputs = tokenizer(prompt, return_tensors="pt").to("cuda")
    with torch.no_grad():
        output = model.generate(**inputs, max_length=max_length, do_sample=True, temperature=0.7, top_p=0.9)
    return tokenizer.decode(output[0], skip_special_tokens=True)

# Read prompt from file
with open("/home/azureuser/LLaMA-Factory/scripts/prompt.txt", "r") as file:
    prompt = file.read().strip()

# Generate response
response = chat(prompt)

with open("/home/azureuser/LLaMA-Factory/scripts/output.txt", "w") as file:
    file.write(response)