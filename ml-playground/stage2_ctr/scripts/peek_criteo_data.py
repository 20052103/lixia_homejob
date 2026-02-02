import os
import yaml

NUM_DENSE = 13
NUM_SPARSE = 26

def detect_delim(line: str) -> str:
    return "\t" if "\t" in line else " "

def main():
    # load config
    cfg = yaml.safe_load(open("configs/local.yaml", "r", encoding="utf-8"))
    data_dir = cfg["data"]["dac_dir"]
    train_file = cfg["data"]["train_file"]
    path = os.path.join(data_dir, train_file)

    print("Reading file:", path)
    print("=" * 80)

    with open(path, "r", encoding="utf-8", errors="ignore") as f:
        first = f.readline()
        delim = detect_delim(first)
        f.seek(0)

        for row_idx in range(3):  # 只看前 3 行
            line = f.readline().strip()
            cols = line.split(delim)

            label = cols[0]
            dense = cols[1 : 1 + NUM_DENSE]
            sparse = cols[1 + NUM_DENSE : 1 + NUM_DENSE + NUM_SPARSE]

            print(f"\n===== ROW {row_idx} =====")
            print("label:", label)

            print("\nDense features (I1–I13):")
            for i, v in enumerate(dense, start=1):
                print(f"  I{i:<2}: {v}")

            print("\nSparse features (C1–C26):")
            for i, v in enumerate(sparse, start=1):
                print(f"  C{i:<2}: {v}")

            print("-" * 80)
            print("\nDense as int:")
        print([int(x) if x != "" else 0 for x in dense])

        print("Dense after log1p:")
        import math
        print([round(math.log1p(int(x)), 4) if x != "" else 0.0 for x in dense])

    def simple_hash(s, seed=0):
        h = 2166136261 ^ seed
        for ch in s:
            h ^= ord(ch)
            h = (h * 16777619) & 0xFFFFFFFF
        return h

    hash_bins = cfg["preprocess"]["hash_bins"]

    print("\nSparse -> hashed embedding ids:")
    print(f"(hash_bins = {hash_bins})")
    for j, v in enumerate(sparse):
        hid = simple_hash(v, seed=j) % hash_bins
        global_id = j * hash_bins + hid
        print(f"  C{j+1:<2}: raw={v:<10}  hashed={hid:<8}  global_id={global_id} j = {j}")

if __name__ == "__main__":
    main()
