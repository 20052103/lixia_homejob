import os
import yaml

def detect_delim(line: str) -> str:
    return "\t" if "\t" in line else " "

def stable_select(line_idx: int, seed: int, val_ratio: float) -> bool:
    # 与你当前 dataset split 一致的稳定选择器（LCG）
    a = 1103515245
    c = 12345
    m = 2**31
    r = (a * (line_idx + seed) + c) % m
    return (r / m) < val_ratio

def main():
    cfg = yaml.safe_load(open("configs/local.yaml", "r", encoding="utf-8"))
    dac_dir = cfg["data"]["dac_dir"]
    train_file = cfg["data"]["train_file"]
    train_path = os.path.normpath(os.path.join(dac_dir, train_file))

    seed = int(cfg["split"]["seed"])
    val_ratio = float(cfg["split"]["val_ratio"])

    out_dir = os.path.normpath(os.path.join(dac_dir, "_gold"))
    os.makedirs(out_dir, exist_ok=True)
    gold_path = os.path.join(out_dir, "gold_eval.txt")

    max_lines = int(cfg.get("gold", {}).get("max_lines", 500_000))  # 可选：限制 gold 大小
    written = 0
    total = 0

    with open(train_path, "r", encoding="utf-8", errors="ignore") as f:
        first = f.readline()
        if not first:
            raise RuntimeError("Empty train file")
        delim = detect_delim(first)
        f.seek(0)

        with open(gold_path, "w", encoding="utf-8") as out:
            for line_idx, line in enumerate(f):
                total += 1
                if stable_select(line_idx, seed=seed, val_ratio=val_ratio):
                    out.write(line)
                    written += 1
                    if written >= max_lines:
                        break

    print("train_path:", train_path)
    print("gold_eval_path:", gold_path)
    print("written_lines:", written, "scanned_lines:", total)
    print("✅ gold eval created.")

if __name__ == "__main__":
    main()
