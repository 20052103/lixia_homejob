import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel, PeftConfig

BASE_MODEL = r"D:\repo\model\models--Qwen--Qwen2.5-7B-Instruct\snapshots\a09a35458c702b33eeacc393d103063234e8bc28"
LORA_PATH  = r"D:\repo\lixia_homejob\llm-rec-interest-qwen\outputs\qwen25_interest_lora_v2"

print("Loading tokenizer...")
tok = AutoTokenizer.from_pretrained(BASE_MODEL, trust_remote_code=True, use_fast=False)

print("Loading base model...")
base = AutoModelForCausalLM.from_pretrained(
    BASE_MODEL, torch_dtype=torch.bfloat16, device_map="auto", trust_remote_code=True
).eval()

print("Reading peft config from adapter...")
cfg = PeftConfig.from_pretrained(LORA_PATH)
print("PEFT type:", cfg.peft_type)
print("Base model in adapter config:", cfg.base_model_name_or_path)
print("Target modules:", getattr(cfg, "target_modules", None))

print("Attaching LoRA...")
lora = PeftModel.from_pretrained(base, LORA_PATH).eval()

# 关键：打印当前激活的 adapter
print("Active adapters:", getattr(lora, "active_adapters", None))

# 关键：统计可训练参数（LoRA 注入后应 >0，且出现 lora_A/lora_B）
names = [n for n,_ in lora.named_parameters()]
lora_params = [n for n in names if "lora_" in n]
print("Num parameters containing 'lora_':", len(lora_params))
print("Example lora param names:", lora_params[:10])

# 再确认一下 forward 是否用的是 PeftModel
print("Model class:", lora.__class__)

# 快速比对 logits：同一输入，base vs lora 的 next-token logits 是否不同
prompt = "<|system|>\nYou are a helpful assistant.\n<|user|>\nTopics: Car, Renault, Ford Focus\n<|assistant|>\n"
inp = tok(prompt, return_tensors="pt").to(base.device)

with torch.no_grad():
    out_base = base(**inp).logits[0, -1, :].float().cpu()
    out_lora = lora(**inp).logits[0, -1, :].float().cpu()

diff = (out_base - out_lora).abs().mean().item()
mx = (out_base - out_lora).abs().max().item()
print(f"Mean abs logits diff (base vs lora): {diff:.8f}")
print(f"Max  abs logits diff (base vs lora): {mx:.8f}")

# 经验判断：如果 diff≈0 且 max≈0，基本就是 LoRA 没起作用（或 adapter 权重全 0）
