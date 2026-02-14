import re
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel

BASE_MODEL = r"D:\repo\model\models--Qwen--Qwen2.5-7B-Instruct\snapshots\a09a35458c702b33eeacc393d103063234e8bc28"
LORA_PATH  = r"D:\repo\lixia_homejob\llm-rec-interest-qwen\outputs\qwen25_interest_lora_v2"

SYSTEM = "You are a helpful assistant."

CASES = [
    (
        "Vehicles + anime + fishing",
        """Here are recent watched video topics (labels). Summarize the user's interests.
1) Vehicle, Car, Renault, Ford Focus
2) Vehicle, Aircraft, Airplane, Airline
3) Cartoon, Inuyasha
4) Fishing, Recreational fishing
""",
    ),
    (
        "Sports + music + cooking",
        """User recently watched videos with these topics. What are their interests?
1) Basketball, NBA, Golden State Warriors
2) Concert, Musician, Drum, Drummer
3) Game, Association football, Premier League
4) Cooking, Recipe, Vegetable
""",
    ),
    (
        "Craft + toys + running + dance",
        """Recent watched topics:
1) Origami, Felt
2) Toy, Lego, Kinder Surprise, Playground
3) Running, Marathon
4) Dance, Performance art
Summarize the user's interests.
""",
    ),
    (
        "Beauty + fashion",
        """Topics watched:
1) Makeup, Cosmetics, Lipstick
2) Hairstyle, Wig
3) Fashion, Streetwear
4) Skincare
Summarize the user's interests.
""",
    ),
]

# Greedy generation for stable comparison
GEN = dict(
    max_new_tokens=140,
    do_sample=False,
    temperature=1.0,
)

STOP_STRINGS = [
    "<|user|>",
    "<|system|>",
    "Human:",
    "\nHuman:",
    "\n<|user|>",
    "\n<|system|>",
]

def build_prompt(user_text: str) -> str:
    return f"<|system|>\n{SYSTEM}\n<|user|>\n{user_text.strip()}\n<|assistant|>\n"

def extract_assistant(decoded: str) -> str:
    # decode with skip_special_tokens=True so no special tokens remain, but keep robust fallback
    if "<|assistant|>" in decoded:
        decoded = decoded.split("<|assistant|>", 1)[1]
    return decoded.strip()

def cut_after_first_answer(text: str) -> str:
    # Cut at known stop strings
    cut = len(text)
    for s in STOP_STRINGS:
        i = text.find(s)
        if i != -1:
            cut = min(cut, i)

    text = text[:cut].strip()

    # Extra: sometimes model continues with Q/A like "Can you..." after a newline without tags
    # Cut if it starts a new question line that looks like a user turn.
    # (Conservative: only cut if it clearly looks like a new turn.)
    lines = text.splitlines()
    kept = []
    for ln in lines:
        # if a line looks like it begins a new user question, stop
        if re.match(r"^\s*(can you|could you|please|why|how)\b", ln.strip(), re.I):
            break
        kept.append(ln)
    return "\n".join(kept).strip() if kept else text

def generate_text(model, tok, prompt: str) -> str:
    inputs = tok(prompt, return_tensors="pt").to(model.device)
    with torch.no_grad():
        out = model.generate(
            **inputs,
            **GEN,
            eos_token_id=tok.eos_token_id,
            pad_token_id=tok.eos_token_id,
        )
    decoded = tok.decode(out[0], skip_special_tokens=True)
    ans = extract_assistant(decoded)
    ans = cut_after_first_answer(ans)
    return ans.strip()

def md_block(title: str, content: str) -> str:
    return f"### {title}\n\n```text\n{content}\n```\n"

def main():
    tok = AutoTokenizer.from_pretrained(BASE_MODEL, trust_remote_code=True, use_fast=False)

    base = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        trust_remote_code=True
    ).eval()

    model = PeftModel.from_pretrained(base, LORA_PATH).eval()

    print("# LoRA OFF vs LoRA ON (Qwen2.5-7B-Instruct)\n")

    for name, user_text in CASES:
        prompt = build_prompt(user_text)

        print(f"## Case: {name}\n")
        print(md_block("Prompt", prompt))

        # LoRA OFF (base behavior)
        with model.disable_adapter():
            off = generate_text(model, tok, prompt)
        print(md_block("LoRA OFF (base behavior)", off))

        # LoRA ON
        on = generate_text(model, tok, prompt)
        print(md_block("LoRA ON", on))

        print("\n---\n")

if __name__ == "__main__":
    main()
