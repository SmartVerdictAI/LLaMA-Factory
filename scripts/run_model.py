from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel
import torch

# Base model
base_model_path = "Qwen/Qwen2.5-72B-Instruct"

# Checkpoint path (update with the latest checkpoint)
checkpoint_path = "/mount/sdb/llama_output/checkpoint-468"

special_tokens = {
    "eos_token": "<|im_end|>",
    "bos_token": "<|im_start|>",
    "pad_token": "<|pad|>"
}

# Load tokenizer
tokenizer = AutoTokenizer.from_pretrained(base_model_path, trust_remote_code=True)
tokenizer.add_special_tokens(special_tokens)

# Load base model
model = AutoModelForCausalLM.from_pretrained(base_model_path, torch_dtype=torch.bfloat16, device_map="auto")

model.resize_token_embeddings(len(tokenizer))

# Load fine-tuned LoRA checkpoint
model = PeftModel.from_pretrained(model, checkpoint_path)

# Move model to GPU
model.eval()

def chat(prompt, max_length=32768):
    inputs = tokenizer(prompt, return_tensors="pt").to("cuda")
    with torch.no_grad():
        output = model.generate(**inputs, max_length=max_length, do_sample=True, temperature=0.7, top_p=0.9, repetition_penalty=1.1, eos_token_id=tokenizer.eos_token_id, pad_token_id=tokenizer.pad_token_id)
    return tokenizer.decode(output[0], skip_special_tokens=True)

# Read prompt from file
with open("/home/azureuser/LLaMA-Factory/scripts/prompt.txt", "r") as file:
    prompt = file.read().strip()

# Generate response
response = chat(prompt)

with open("/home/azureuser/LLaMA-Factory/scripts/output_tokentest6.txt", "w") as file:
    file.write(response)


# input_text = "<|im_start|>user\nTell me a joke.<|im_end|>\n<|im_start|>assistant"
# input_ids = tokenizer(input_text, return_tensors="pt").input_ids.to("cuda")

# output = model.generate(
#     input_ids,
#     max_length=50,
#     do_sample=True,
#     temperature=0.7,
#     top_p=0.9,
#     repetition_penalty=1.2,
#     eos_token_id=tokenizer.eos_token_id,  # Ensure it stops at <|im_end|>
#     pad_token_id=tokenizer.pad_token_id   # Prevents weird outputs
# )

# response = tokenizer.decode(output[0], skip_special_tokens=False)
# print(response)
