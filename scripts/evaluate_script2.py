from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel
import torch
import os

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
# model = PeftModel.from_pretrained(model, checkpoint_path)

# Move model to GPU
model.eval()

def chat(prompt, max_length=32768):
    inputs = tokenizer(prompt, return_tensors="pt").to("cuda")
    with torch.no_grad():
        output = model.generate(**inputs, max_length=max_length, do_sample=True, temperature=0.7, top_p=0.9, repetition_penalty=1.1, eos_token_id=tokenizer.eos_token_id, pad_token_id=tokenizer.pad_token_id)
    return tokenizer.decode(output[0], skip_special_tokens=True)

def make_prompt(text):
    return f"""<|im_start|>system
书写本案的仲裁庭意见，必须把以下要求写到本案的仲裁庭意见书中：
1、在书写本案的仲裁庭意见时，要保证逻辑的清晰，要充分根据本案案件资料内容，具体根据申请人的仲裁请求进行回应：具体体现在申请人的仲裁请求+案件的事实依据、证据体现+被申请人的质证意见/被申请人的出庭情况+法律规定的内容（要写出具体的法条）+仲裁庭对于该请求的具体意见，以上信息在书写仲裁庭意见内容时非常非常重要，，务必务必重视。 2、在书写本案的仲裁庭意见时，绝对不能遗漏申请人的全部仲裁请求，全部仲裁请求都要写到仲裁庭意见中，针对于申请人的具体仲裁请求的小标题，要简要书写，不能照抄原文；其中，《关于申请人要求被申请人承担仲裁费用的请求；》这条仲裁请求，单独写到最后 3、另查明：后的内容一般为律师费、保理费、经营范围等，具体可参照示例案件资料的书写逻辑书写。 4、在书写本案的仲裁庭意见时，只要是涉及到事实依据、证据内容的，要把内容写的足够清晰，可以参考给你的示例案件资料的书写逻辑 5、关于合同效力，要能够清晰完整的表达本案的全部事实依据，如果在笔录中申请人/被申请人有在当庭对合同的真实性无异议，或者是说他承认自己对所签的合同是自己签的类似的笔录，这种类似的表述可以加到关于合同的效力的内容里面。 6、不需要写与仲裁庭意见无关的内容 7、请写出四、仲裁庭意见和五、裁决部分。
<|im_end|>

<|im_start|>user
{text}
<|im_end|>

<|im_start|>assistant"""


def evaluate_benchmark(input_dir, output_dir):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    for filename in os.listdir(input_dir):
        input_path = os.path.join(input_dir, filename)
        output_path = os.path.join(output_dir, filename)

        with open(input_path, "r") as file:
            prompt = file.read().strip()

        prompt = make_prompt(prompt)
        response = chat(prompt)

        with open(output_path, "w") as file:
            file.write(response)

        print(f"Generated response for {filename}")


# # Read prompt from file
# with open("/home/azureuser/LLaMA-Factory/scripts/prompt.txt", "r") as file:
#     prompt = file.read().strip()

# # Generate response
# response = chat(prompt)

# with open("/home/azureuser/LLaMA-Factory/scripts/output_tokentest6.txt", "w") as file:
#     file.write(response)


input_dir = "/home/azureuser/LLaMA-Factory/scripts/input_files"
output_dir = "/home/azureuser/LLaMA-Factory/scripts/evaluations_base"

evaluate_benchmark(input_dir, output_dir)