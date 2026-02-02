import os
import math
import yaml
from collections import Counter, defaultdict

NUM_DENSE = 13
NUM_SPARSE = 26

def detect_delim(line: str) -> str:
    return "\t" if ("\t" in line) else " "

def safe_int(x: str):
    if x is None or x == "":
        return None
    try:
        return int(x)
    except ValueError:
        return None

def main():
    cfg = yaml.safe_load(open("configs/local.yaml", "r", encoding="utf-8"))
    dac_dir = cfg["data"]["dac_dir"]
    train_file = cfg["data"]["train_file"]
    train_path = os.path.normpath(os.path.join(dac_dir, train_file))

    # 你可以改：抽样多少行
    max_lines = int(cfg.get("inspect", {}).get("max_lines", 200_000))

    dense_names = [f"I{i}" for i in range(1, NUM_DENSE + 1)]
    sparse_names = [f"C{i}" for i in range(1, NUM_SPARSE + 1)]

    # dense stats
    dense_min = [math.inf] * NUM_DENSE
    dense_max = [-math.inf] * NUM_DENSE
    dense_sum = [0.0] * NUM_DENSE
    dense_cnt = [0] * NUM_DENSE
    dense_missing = [0] * NUM_DENSE

    # sparse stats (抽样统计 raw value)
    sparse_missing = [0] * NUM_SPARSE
    sparse_cnt = [0] * NUM_SPARSE
    # 只保留 TopK 计数（避免内存爆）；这里用 Counter 但会限制 size
    topk_limit = int(cfg.get("inspect", {}).get("sparse_topk_limit", 50))
    sparse_top = [Counter() for _ in range(NUM_SPARSE)]
    sparse_unique_est = [set() for _ in range(NUM_SPARSE)]  # 仅用于小样本估计（max_lines <= 2e5 ok）

    pos = 0
    total = 0

    with open(train_path, "r", encoding="utf-8", errors="ignore") as f:
        first = f.readline()
        if not first:
            raise RuntimeError("Empty train file")
        delim = detect_delim(first)
        f.seek(0)

        for i, line in enumerate(f):
            if i >= max_lines:
                break
            line = line.rstrip("\n")
            if not line:
                continue
            cols = line.split(delim)
            if len(cols) < 1 + NUM_DENSE:
                continue

            # label
            y = safe_int(cols[0])
            if y is not None and y > 0:
                pos += 1
            total += 1

            dense_raw = cols[1 : 1 + NUM_DENSE]
            sparse_raw = cols[1 + NUM_DENSE : 1 + NUM_DENSE + NUM_SPARSE]
            if len(sparse_raw) < NUM_SPARSE:
                sparse_raw = sparse_raw + [""] * (NUM_SPARSE - len(sparse_raw))

            # dense
            for j, x in enumerate(dense_raw):
                v = safe_int(x)
                if v is None:
                    dense_missing[j] += 1
                else:
                    dense_cnt[j] += 1
                    dense_sum[j] += float(v)
                    if v < dense_min[j]:
                        dense_min[j] = v
                    if v > dense_max[j]:
                        dense_max[j] = v

            # sparse
            for j, x in enumerate(sparse_raw[:NUM_SPARSE]):
                if x is None or x == "":
                    sparse_missing[j] += 1
                else:
                    sparse_cnt[j] += 1
                    # topk
                    sparse_top[j][x] += 1
                    # unique (sample-based)
                    if len(sparse_unique_est[j]) < 200_000:  # cap
                        sparse_unique_est[j].add(x)

    if total == 0:
        raise RuntimeError("No valid lines parsed. Delimiter or file format may be wrong.")

    print("=" * 80)
    print("Criteo DAC schema (no header in file, using conventional names)")
    print("File:", train_path)
    print("Sample lines:", total)
    print(f"Label positive rate (sample): {pos/total:.6f}  (pos={pos}, total={total})")
    print("=" * 80)

    print("\n[DENSE FEATURES]")
    print("name\tmissing_rate\tcount\tmin\tmax\tmean")
    for j, name in enumerate(dense_names):
        miss_rate = dense_missing[j] / total
        cnt = dense_cnt[j]
        if cnt > 0:
            mean = dense_sum[j] / cnt
            mn = dense_min[j] if dense_min[j] != math.inf else None
            mx = dense_max[j] if dense_max[j] != -math.inf else None
        else:
            mean, mn, mx = None, None, None
        print(f"{name}\t{miss_rate:.4f}\t{cnt}\t{mn}\t{mx}\t{mean}")

    print("\n[SPARSE FEATURES]")
    print("name\tmissing_rate\tnon_empty\tunique_est(sample)\ttop_values(sample)")
    for j, name in enumerate(sparse_names):
        miss_rate = sparse_missing[j] / total
        non_empty = sparse_cnt[j]
        uniq = len(sparse_unique_est[j])
        top = sparse_top[j].most_common(topk_limit)
        top_str = ", ".join([f"{k}:{v}" for k, v in top[:10]])  # 只打印前10个最频繁
        print(f"{name}\t{miss_rate:.4f}\t{non_empty}\t{uniq}\t{top_str}")

if __name__ == "__main__":
    main()
