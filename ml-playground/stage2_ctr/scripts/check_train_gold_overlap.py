import hashlib

train_path = r"D:\repo\large data\criteo\dac\train.txt"
gold_path  = r"D:\repo\large data\criteo\dac\_gold\gold_eval.txt"

def line_hash(line: str) -> str:
    return hashlib.md5(line.rstrip("\n").encode("utf-8")).hexdigest()

print("Loading train hashes...")
train_hashes = set()
with open(train_path, "r", encoding="utf-8", errors="ignore") as f:
    for i, line in enumerate(f):
        train_hashes.add(line_hash(line))
        if (i + 1) % 1_000_000 == 0:
            print(f"  loaded {i+1} train lines")

print("Checking gold overlap...")
overlap = 0
with open(gold_path, "r", encoding="utf-8", errors="ignore") as f:
    for line in f:
        if line_hash(line) in train_hashes:
            overlap += 1

print("\n===== OVERLAP CHECK =====")
print("overlap_lines:", overlap)
print("status:", "OK (no leakage)" if overlap == 0 else "WARNING (data leakage!)")
