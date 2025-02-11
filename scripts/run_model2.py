from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
import deepspeed
from peft import PeftModel

base_model_path = "Qwen/Qwen2.5-72B-Instruct"
checkpoint_path = "/mount/sdb/llama_output/checkpoint-468"

# Load tokenizer
tokenizer = AutoTokenizer.from_pretrained(base_model_path, trust_remote_code=True)

# Enable DeepSpeed (Optimized)
ds_config = {
  "train_batch_size": "auto",
  "train_micro_batch_size_per_gpu": "auto",
  "gradient_accumulation_steps": "auto",
  "gradient_clipping": "auto",
  "zero_allow_untested_optimizer": True,
  "fp16": {
    "enabled": "auto",
    "loss_scale": 0,
    "loss_scale_window": 1000,
    "initial_scale_power": 16,
    "hysteresis": 2,
    "min_loss_scale": 1
  },
  "bf16": {
    "enabled": "auto"
  },
  "zero_optimization": {
    "stage": 3,
    "overlap_comm": True,
    "contiguous_gradients": True,
    "sub_group_size": 1e9,
    "reduce_bucket_size": "auto",
    "stage3_prefetch_bucket_size": "auto",
    "stage3_param_persistence_threshold": "auto",
    "stage3_max_live_parameters": 1e9,
    "stage3_max_reuse_distance": 1e9,
    "stage3_gather_16bit_weights_on_model_save": True
  }
}


# Load base model
model = AutoModelForCausalLM.from_pretrained(base_model_path, torch_dtype=torch.float16, device_map="auto")

# Apply DeepSpeed *before* loading the PEFT adapter
model = deepspeed.init_inference(model, config=ds_config)

# Load PEFT adapter (LoRA Checkpoint)
model = PeftModel.from_pretrained(model.module if hasattr(model, 'module') else model, checkpoint_path)

# Ensure model is in evaluation mode
model.eval().cuda()

# Chat function
def chat(prompt, max_length=4096):  # Reduce max_length to avoid OOM
    inputs = tokenizer(prompt, return_tensors="pt").to("cuda")
    with torch.no_grad():
        output = model.generate(**inputs, max_length=max_length, do_sample=True, temperature=0.7, top_p=0.9)
    return tokenizer.decode(output[0], skip_special_tokens=True)

# Read prompt from file
with open("/home/azureuser/LLaMA-Factory/scripts/prompt.txt", "r") as file:
    prompt = file.read().strip()

# Generate response
response = chat(prompt)

# Write output to file
with open("/home/azureuser/LLaMA-Factory/scripts/output.txt", "w") as file:
    file.write(response)