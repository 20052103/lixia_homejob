import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel

BASE_MODEL = r"D:\repo\model\models--Qwen--Qwen2.5-7B-Instruct\snapshots\a09a35458c702b33eeacc393d103063234e8bc28"
LORA_PATH  = r"D:\repo\lixia_homejob\llm-rec-interest-qwen\outputs\qwen25_interest_lora_v2"

SYSTEM = "You are a helpful assistant."

USER = """Here are recent watched video topics (labels). Summarize the user's interests.
1) Vehicle, Car, Renault, Ford Focus
2) Vehicle, Aircraft, Airplane, Airline
3) Cartoon, Inuyasha
4) Fishing, Recreational fishing
""".strip()

PROMPT = f"<|system|>\n{SYSTEM}\n<|user|>\n{USER}\n<|assistant|>\n"

GEN = dict(
    max_new_tokens=120,
    do_sample=False,          # 用 greedy，差异更稳定
    temperature=1.0,
)

def run(model, tok, prompt):
    inputs = tok(prompt, return_tensors="pt").to(model.device)
    with torch.no_grad():
        out = model.generate(
            **inputs,
            max_new_tokens=120,
            do_sample=False,
            eos_token_id=tok.eos_token_id,
            pad_token_id=tok.eos_token_id,
            stop_strings=["<|user|>", "<|system|>", "Human:", "\n<|user|>"],
            tokenizer=tok,
        )

    txt = tok.decode(out[0], skip_special_tokens=True)
    if "<|assistant|>" in txt:
        txt = txt.split("<|assistant|>", 1)[1]
    return txt.strip()

def main():
    tok = AutoTokenizer.from_pretrained(BASE_MODEL, trust_remote_code=True, use_fast=False)

    base = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL, torch_dtype=torch.bfloat16, device_map="auto", trust_remote_code=True
    ).eval()

    model = PeftModel.from_pretrained(base, LORA_PATH).eval()

    print("\n" + "="*100)
    print("PROMPT")
    print("="*100)
    print(PROMPT)

    print("\n" + "="*100)
    print("LoRA OFF (base behavior)")
    print("="*100)
    with model.disable_adapter():
        print(run(model, tok, PROMPT))

    print("\n" + "="*100)
    print("LoRA ON")
    print("="*100)
    print(run(model, tok, PROMPT))

if __name__ == "__main__":
    main()
