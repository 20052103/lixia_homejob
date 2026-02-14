import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel

# ===== PATHS =====
BASE_MODEL = r"D:\repo\model\models--Qwen--Qwen2.5-7B-Instruct\snapshots\a09a35458c702b33eeacc393d103063234e8bc28"
LORA_PATH = r"D:\repo\lixia_homejob\llm-rec-interest-qwen\outputs\qwen25_interest_lora"

device = "cuda"

# ===== Load tokenizer =====
tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL, trust_remote_code=True, use_fast=False)

# ===== Load base model =====
model = AutoModelForCausalLM.from_pretrained(
    BASE_MODEL,
    torch_dtype=torch.bfloat16,
    device_map="auto",
    trust_remote_code=True
)

# ===== Attach LoRA adapter =====
model = PeftModel.from_pretrained(model, LORA_PATH)
model.eval()

# ===== Example pseudo-user input =====
SYSTEM = "You are a user interest summarization assistant. Summarize the user's interests clearly and concisely."

USER_INPUT = """
Summarize the user's interests based on their recent watched videos.

Recent watched video topics (auto-extracted labels):
1. Labels: Vehicle, Car, Renault, Ford Focus
2. Labels: Vehicle, Aircraft, Airplane, Airline
3. Labels: Cartoon, Inuyasha
4. Labels: Fishing, Recreational fishing
"""

prompt = (
    "<|system|>\n" + SYSTEM + "\n"
    "<|user|>\n" + USER_INPUT + "\n"
    "<|assistant|>\n"
)

inputs = tokenizer(prompt, return_tensors="pt").to(device)

with torch.no_grad():
    outputs = model.generate(
        **inputs,
        max_new_tokens=150,
        temperature=0.7,
        top_p=0.9,
        do_sample=True
    )

decoded = tokenizer.decode(outputs[0], skip_special_tokens=True)

print("\n===== MODEL OUTPUT =====\n")
print(decoded)
