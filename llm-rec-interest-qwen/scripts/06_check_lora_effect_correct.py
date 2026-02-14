import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel

BASE_MODEL = r"D:\repo\model\models--Qwen--Qwen2.5-7B-Instruct\snapshots\a09a35458c702b33eeacc393d103063234e8bc28"
LORA_PATH  = r"D:\repo\lixia_homejob\llm-rec-interest-qwen\outputs\qwen25_interest_lora_v2"

tok = AutoTokenizer.from_pretrained(BASE_MODEL, trust_remote_code=True, use_fast=False)

base = AutoModelForCausalLM.from_pretrained(
    BASE_MODEL, torch_dtype=torch.bfloat16, device_map="auto", trust_remote_code=True
).eval()

model = PeftModel.from_pretrained(base, LORA_PATH).eval()

prompt = "<|system|>\nYou are a helpful assistant.\n<|user|>\nTopics: Car, Renault, Ford Focus\n<|assistant|>\n"
inp = tok(prompt, return_tensors="pt").to(model.device)

with torch.no_grad():
    # LoRA enabled (default)
    logits_on = model(**inp).logits[0, -1, :].float().cpu()

    # LoRA disabled
    with model.disable_adapter():
        logits_off = model(**inp).logits[0, -1, :].float().cpu()

diff_mean = (logits_on - logits_off).abs().mean().item()
diff_max  = (logits_on - logits_off).abs().max().item()

print("Mean abs logits diff (LoRA ON vs OFF):", f"{diff_mean:.8f}")
print("Max  abs logits diff (LoRA ON vs OFF):", f"{diff_max:.8f}")
