import os
import math

def detect_delim(line: str) -> str:
    return "\t" if "\t" in line else " "

def main():
    # ✅ 改成你的真实路径
    train_path = r"D:\repo\large data\criteo\dac\train.txt"
    max_lines = 200_000  # 抽样行数

    pos = 0
    total = 0

    with open(train_path, "r", encoding="utf-8", errors="ignore") as f:
        first = f.readline()
        if not first:
            raise RuntimeError("Empty file.")
        delim = detect_delim(first)
        f.seek(0)

        for i, line in enumerate(f):
            if i >= max_lines:
                break
            line = line.rstrip("\n")
            if not line:
                continue
            cols = line.split(delim)
            if len(cols) < 1:
                continue
            y = cols[0]
            if y == "1":
                pos += 1
            total += 1

    if total == 0:
        raise RuntimeError("No valid lines parsed.")

    p = pos / total
    # baseline: always predict probability = p
    baseline_logloss = -(p * math.log(p + 1e-12) + (1 - p) * math.log(1 - p + 1e-12))

    print("train_path:", train_path)
    print("sample_lines:", total)
    print("pos:", pos)
    print("p (positive rate):", p)
    print("baseline_logloss (always predict p):", baseline_logloss)

if __name__ == "__main__":
    main()
