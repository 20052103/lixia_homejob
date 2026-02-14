import os
import json
import random
import csv
from collections import Counter
import tensorflow as tf


# =======================
# Paths (edit if needed)
# =======================
YT8M_ROOT = r"D:\repo\large data\yt8m"
TFRECORD_DIR = os.path.join(YT8M_ROOT, "video_level")
VOCAB_CSV = os.path.join(YT8M_ROOT, "vocabulary.csv")

TRAIN_TFRECORD = os.path.join(TFRECORD_DIR, "train00.tfrecord")
VALID_TFRECORD = os.path.join(TFRECORD_DIR, "validate00.tfrecord")

OUT_DIR = r"D:\repo\lixia_homejob\llm-rec-interest-qwen\data\processed"
OUT_PATH = os.path.join(OUT_DIR, "yt8m_interest_sft.jsonl")


# =======================
# Vocabulary: label_id(Index) -> Name
# =======================
def load_vocab(vocab_csv_path: str) -> dict[int, str]:
    """
    vocabulary.csv header (your file):
    Index,TrainVideoCount,KnowledgeGraphId,Name,WikiUrl,Vertical1,Vertical2,Vertical3,WikiDescription

    We want:
      label_id (Index) -> Name (human readable)
    """
    vocab: dict[int, str] = {}
    with open(vocab_csv_path, "r", encoding="utf-8", newline="") as f:
        reader = csv.reader(f)
        header = next(reader, None)  # skip header
        for row in reader:
            if not row or len(row) < 4:
                continue
            try:
                label_id = int(row[0])  # Index
            except Exception:
                continue
            name = row[3].strip()  # Name
            if name:
                vocab[label_id] = name

    if not vocab:
        raise RuntimeError(f"Failed to parse vocab Name column: {vocab_csv_path}")
    return vocab


# =======================
# TFRecord iterator (video-level)
# =======================
def iter_yt8m_tfrecord(tfrecord_path: str):
    feature_desc = {
        "id": tf.io.FixedLenFeature([], tf.string),
        "labels": tf.io.VarLenFeature(tf.int64),
    }
    for raw in tf.data.TFRecordDataset([tfrecord_path]):
        ex = tf.io.parse_single_example(raw, feature_desc)
        vid = ex["id"].numpy().decode("utf-8", errors="ignore")
        labels = tf.sparse.to_dense(ex["labels"]).numpy().tolist()
        yield vid, labels


# =======================
# Output template (EN)
# =======================
def make_output(top_labels: list[str]) -> str:
    """
    Stronger target:
      - Primary interests
      - Secondary interests
      - Recommendation direction

    Important: do NOT over-filter here to empty.
    We'll filter some too-general labels, but always fallback.
    """
    # Too-generic labels (feel free to tweak)
    stop = {
        "Game",
        "Vehicle",
        "Music",
        "Food",
        "Toy",
        "Cartoon",
        "Comedy (drama)",
        "Performance art",
        "Outdoor recreation",
        "Concert",
    }

    cleaned = []
    for x in top_labels:
        x = x.strip()
        if not x:
            continue
        if x.startswith("label_"):
            continue
        if x in stop:
            continue
        # filter very short noise
        if len(x) <= 2:
            continue
        cleaned.append(x)

    # fallback if over-filtered
    if len(cleaned) < 6:
        cleaned = [x.strip() for x in top_labels if x.strip() and not x.startswith("label_")][:10]
    else:
        cleaned = cleaned[:10]

    primary = cleaned[:3]
    secondary = cleaned[3:7]

    # Simple rec direction
    rec_parts = []
    if len(primary) >= 2:
        rec_parts.append(f"more content related to {primary[0]} and {primary[1]}")
    elif len(primary) == 1:
        rec_parts.append(f"more content related to {primary[0]}")

    if len(secondary) >= 2:
        rec_parts.append(f"occasional videos about {secondary[0]} and {secondary[1]}")
    elif len(secondary) == 1:
        rec_parts.append(f"occasional videos about {secondary[0]}")

    rec_text = "; ".join(rec_parts) if rec_parts else "more content aligned with these interests"

    return (
        f"Primary interests: {', '.join(primary)}.\n"
        f"Secondary interests: {', '.join(secondary)}.\n"
        f"Recommendation direction: {rec_text}."
    )


# =======================
# Build SFT samples (pseudo users)
# =======================
def build_sft_samples(
    videos: list[list[str]],
    seed: int = 42,
    videos_per_user: int = 8,
    topn_labels: int = 12,
    num_users: int = 2000,
):
    """
    videos: list of per-video label-name lists
    pseudo-user: sample K videos -> aggregate labels -> produce an interest summary
    """
    random.seed(seed)
    n = len(videos)
    if n == 0:
        raise RuntimeError("No videos loaded; check TFRecord parsing.")

    instruction = "Summarize the user's interests based on their recent watched videos."
    samples = []

    for _ in range(num_users):
        chosen = [videos[random.randrange(n)] for _ in range(videos_per_user)]

        # IMPORTANT: in aggregation, DO NOT remove generic stopwords.
        # Only remove missing/label_ noise. Stopwords are handled in make_output().
        flat = []
        for labs in chosen:
            for lab in labs:
                if not lab or lab == "nan":
                    continue
                if lab.startswith("label_"):
                    continue
                flat.append(lab)

        c = Counter(flat)
        if not c:
            continue

        top_labels = [k for k, _ in c.most_common(topn_labels)]

        # Build input text
        lines = ["Recent watched video topics (auto-extracted labels):"]
        for i, labs in enumerate(chosen, 1):
            labs2 = [x for x in labs if x and x != "nan" and not x.startswith("label_")][:6]
            if not labs2:
                labs2 = ["(noisy_labels_filtered)"]
            lines.append(f"{i}. Labels: {', '.join(labs2)}")

        samples.append(
            {
                "instruction": instruction,
                "input": "\n".join(lines),
                "output": make_output(top_labels),
            }
        )

    return samples


# =======================
# Main
# =======================
def main():
    os.makedirs(OUT_DIR, exist_ok=True)

    vocab = load_vocab(VOCAB_CSV)

    # Load video label lists
    videos: list[list[str]] = []
    for path in [TRAIN_TFRECORD, VALID_TFRECORD]:
        if not os.path.exists(path):
            print(f"Skip missing: {path}")
            continue

        for _, label_ids in iter_yt8m_tfrecord(path):
            names = [vocab.get(int(i), f"label_{int(i)}") for i in label_ids]

            # unique per video, preserve order
            seen = set()
            uniq = []
            for x in names:
                if x not in seen:
                    seen.add(x)
                    uniq.append(x)

            # keep even if it contains generic labels; filtering happens later
            videos.append(uniq)

    print(f"Loaded videos: {len(videos)}")

    samples = build_sft_samples(
        videos,
        seed=42,
        videos_per_user=8,
        topn_labels=12,
        num_users=2000,
    )

    print(f"SFT samples: {len(samples)}")

    with open(OUT_PATH, "w", encoding="utf-8") as f:
        for s in samples:
            f.write(json.dumps(s, ensure_ascii=False) + "\n")

    print(f"Wrote: {OUT_PATH}")


if __name__ == "__main__":
    main()
