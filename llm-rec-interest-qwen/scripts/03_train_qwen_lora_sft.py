import os
import torch
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForCausalLM, TrainingArguments, Trainer, DataCollatorForLanguageModeling
from peft import LoraConfig, get_peft_model

# ===== Paths =====
MODEL_DIR = r"D:\repo\model\models--Qwen--Qwen2.5-7B-Instruct\snapshots\a09a35458c702b33eeacc393d103063234e8bc28"
DATA_PATH = r"D:\repo\lixia_homejob\llm-rec-interest-qwen\data\processed\yt8m_chatml_train.jsonl"
OUTPUT_DIR = r"D:\repo\lixia_homejob\llm-rec-interest-qwen\outputs\qwen25_interest_lora"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# ===== Dataset =====
ds = load_dataset("json", data_files=DATA_PATH, split="train")

# ===== Tokenizer =====
tokenizer = AutoTokenizer.from_pretrained(MODEL_DIR, trust_remote_code=True, use_fast=False)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

MAX_LEN = 1024

def tokenize(ex):
    # ex["text"] is already ChatML with <|assistant|> ... answer
    out = tokenizer(
        ex["text"],
        truncation=True,
        max_length=MAX_LEN,
        padding=False,
    )
    # Causal LM: labels = input_ids
    out["labels"] = out["input_ids"].copy()
    return out

ds_tok = ds.map(tokenize, remove_columns=ds.column_names)

# ===== Model =====
model = AutoModelForCausalLM.from_pretrained(
    MODEL_DIR,
    torch_dtype=torch.bfloat16,
    device_map="auto",
    trust_remote_code=True
)

# ===== LoRA =====
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

# ===== Collator =====
collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)

# ===== Train args =====
args = TrainingArguments(
    output_dir=OUTPUT_DIR,
    per_device_train_batch_size=1,
    gradient_accumulation_steps=16,
    num_train_epochs=2,
    learning_rate=2e-4,
    warmup_ratio=0.03,
    lr_scheduler_type="cosine",
    logging_steps=10,
    save_steps=200,
    save_total_limit=2,
    bf16=True,
    fp16=False,
    report_to="none",
)

trainer = Trainer(
    model=model,
    args=args,
    train_dataset=ds_tok,
    data_collator=collator,
)

trainer.train()

# Save LoRA adapter + tokenizer
model.save_pretrained(OUTPUT_DIR)
tokenizer.save_pretrained(OUTPUT_DIR)

print("✅ Training done. LoRA adapter saved to:", OUTPUT_DIR)
