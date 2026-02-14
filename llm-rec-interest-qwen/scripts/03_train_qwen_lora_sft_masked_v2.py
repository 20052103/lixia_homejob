import os
import torch
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForCausalLM, TrainingArguments, Trainer
from peft import LoraConfig, get_peft_model

BASE_MODEL = r"D:\repo\model\models--Qwen--Qwen2.5-7B-Instruct\snapshots\a09a35458c702b33eeacc393d103063234e8bc28"
DATA_PATH  = r"D:\repo\lixia_homejob\llm-rec-interest-qwen\data\processed\yt8m_chatml_train.jsonl"
OUTPUT_DIR = r"D:\repo\lixia_homejob\llm-rec-interest-qwen\outputs\qwen25_interest_lora_v2"
os.makedirs(OUTPUT_DIR, exist_ok=True)

ds = load_dataset("json", data_files=DATA_PATH, split="train")

tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL, trust_remote_code=True, use_fast=False)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

MAX_LEN = 1024
ASSIST_TAG = "<|assistant|>\n"

def tokenize_mask(ex):
    text = ex["text"]
    tok = tokenizer(text, truncation=True, max_length=MAX_LEN, padding=False)
    input_ids = tok["input_ids"]

    labels = [-100] * len(input_ids)

    idx = text.find(ASSIST_TAG)
    if idx == -1:
        labels = input_ids.copy()
    else:
        prefix = text[: idx + len(ASSIST_TAG)]
        prefix_ids = tokenizer(prefix, truncation=True, max_length=MAX_LEN, padding=False)["input_ids"]
        start = min(len(prefix_ids), len(input_ids))
        labels[start:] = input_ids[start:]

    tok["labels"] = labels
    return tok

ds_tok = ds.map(tokenize_mask, remove_columns=ds.column_names)

model = AutoModelForCausalLM.from_pretrained(
    BASE_MODEL,
    torch_dtype=torch.bfloat16,
    device_map="auto",
    trust_remote_code=True
)

lora = LoraConfig(
    r=16,
    lora_alpha=32,
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM",
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
)
model = get_peft_model(model, lora)
model.print_trainable_parameters()

def collate(batch):
    max_len = max(len(x["input_ids"]) for x in batch)

    def pad(seq, pad_id):
        return seq + [pad_id] * (max_len - len(seq))

    input_ids = torch.tensor([pad(x["input_ids"], tokenizer.pad_token_id) for x in batch], dtype=torch.long)
    attention_mask = torch.tensor([pad(x["attention_mask"], 0) for x in batch], dtype=torch.long)
    labels = torch.tensor([pad(x["labels"], -100) for x in batch], dtype=torch.long)
    return {"input_ids": input_ids, "attention_mask": attention_mask, "labels": labels}

args = TrainingArguments(
    output_dir=OUTPUT_DIR,
    per_device_train_batch_size=1,
    gradient_accumulation_steps=16,
    num_train_epochs=3,
    learning_rate=2e-4,
    lr_scheduler_type="cosine",
    warmup_ratio=0.03,
    logging_steps=10,
    save_steps=200,
    save_total_limit=2,
    bf16=True,
    fp16=False,
    report_to="none",
)

trainer = Trainer(model=model, args=args, train_dataset=ds_tok, data_collator=collate)
trainer.train()

model.save_pretrained(OUTPUT_DIR)
tokenizer.save_pretrained(OUTPUT_DIR)
print("✅ v2 masked SFT done. Adapter saved to:", OUTPUT_DIR)
