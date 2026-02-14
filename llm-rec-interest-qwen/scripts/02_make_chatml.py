import json
import os

IN_PATH = r"D:\repo\lixia_homejob\llm-rec-interest-qwen\data\processed\yt8m_interest_sft.jsonl"
OUT_PATH = r"D:\repo\lixia_homejob\llm-rec-interest-qwen\data\processed\yt8m_chatml_train.jsonl"

SYSTEM = "You are a user interest summarization assistant. Summarize the user's interests clearly and concisely."

def to_chatml(rec):
    user = rec["instruction"].strip()
    if rec.get("input", "").strip():
        user += "\n" + rec["input"].strip()
    assistant = rec["output"].strip()

    text = (
        "<|system|>\n" + SYSTEM + "\n"
        "<|user|>\n" + user + "\n"
        "<|assistant|>\n" + assistant
    )
    return {"text": text}

def main():
    os.makedirs(os.path.dirname(OUT_PATH), exist_ok=True)
    n = 0
    with open(IN_PATH, "r", encoding="utf-8") as f_in, open(OUT_PATH, "w", encoding="utf-8") as f_out:
        for line in f_in:
            rec = json.loads(line)
            f_out.write(json.dumps(to_chatml(rec), ensure_ascii=False) + "\n")
            n += 1
    print(f"Wrote {n} lines to {OUT_PATH}")

if __name__ == "__main__":
    main()
