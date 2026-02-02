import os

# -------- config --------
SRC = os.path.join("data", "train.txt")      # original Criteo train
DST = os.path.join("data", "train_5m.txt")   # subset output
N_LINES = 5_000_000                           # number of lines to keep
# ------------------------

def main():
    if not os.path.exists(SRC):
        raise FileNotFoundError(
            f"Not found: {SRC}\n"
            "Please make sure Criteo train.txt is under stage2_ctr/data/train.txt"
        )

    print("Reading :", SRC)
    print("Writing :", DST)
    print("Lines   :", f"{N_LINES:,}")

    with open(SRC, "r", encoding="utf-8", errors="ignore") as fin, \
         open(DST, "w", encoding="utf-8") as fout:
        for i, line in enumerate(fin):
            if i >= N_LINES:
                break

            fout.write(line)

            if (i + 1) % 1_000_000 == 0:
                print(f"Wrote {i + 1:,} lines")

    print("Done ✅")

if __name__ == "__main__":
    main()
