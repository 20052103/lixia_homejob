import os
import sys
import json
import argparse

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

# allow: python .\scripts\eval_on_gold_dcn_v2.py
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from models.dcn_v2_ctr import DCNV2CTR
from datasets.criteo_dac import CriteoDACIterable, DACHParams

try:
    import yaml
except ImportError:
    yaml = None


def fast_auc(y_true, y_score) -> float:
    y_true = torch.tensor(y_true, dtype=torch.float64)
    y_score = torch.tensor(y_score, dtype=torch.float64)

    pos = (y_true == 1).sum().item()
    neg = (y_true == 0).sum().item()
    if pos == 0 or neg == 0:
        return float("nan")

    sorted_score, idx = torch.sort(y_score)
    sorted_true = y_true[idx]

    n = len(sorted_score)
    ranks = torch.empty(n, dtype=torch.float64)
    i = 0
    rank = 1
    while i < n:
        j = i
        while j + 1 < n and sorted_score[j + 1] == sorted_score[i]:
            j += 1
        avg = (rank + (rank + (j - i))) / 2.0
        ranks[i : j + 1] = avg
        rank += (j - i + 1)
        i = j + 1

    sum_ranks_pos = ranks[sorted_true == 1].sum().item()
    auc = (sum_ranks_pos - pos * (pos + 1) / 2.0) / (pos * neg)
    return float(auc)


@torch.no_grad()
def eval_model(model, loader, device, max_batches=None):
    model.eval()
    bce_sum = 0.0
    n = 0
    all_probs = []
    all_y = []

    for bi, (dense, sparse, y) in enumerate(loader):
        if max_batches is not None and bi >= max_batches:
            break

        dense = dense.to(device, non_blocking=True)
        sparse = sparse.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)

        logits = model(dense, sparse)
        loss = nn.functional.binary_cross_entropy_with_logits(logits, y, reduction="sum")
        bce_sum += loss.item()
        n += y.numel()

        probs = torch.sigmoid(logits).detach().cpu().tolist()
        ys = y.detach().cpu().tolist()
        all_probs.extend(probs)
        all_y.extend(ys)

    logloss = bce_sum / max(1, n)
    pos_rate = float(sum(all_y) / max(1, len(all_y)))
    auc = fast_auc(all_y, all_probs)
    avg_pred = float(sum(all_probs) / max(1, len(all_probs)))

    return {
        "logloss": float(logloss),
        "auc": float(auc),
        "pos_rate": pos_rate,
        "avg_pred": avg_pred,
        "num_samples": int(len(all_y)),
    }


def load_cfg(path):
    if not os.path.exists(path):
        return None
    if yaml is None:
        raise RuntimeError("PyYAML not installed. Run: pip install pyyaml")
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model_dir", type=str, required=True)
    ap.add_argument("--gold_path", type=str, default=r"D:\repo\large data\criteo\dac\_gold\gold_eval.txt")
    ap.add_argument("--batch_size", type=int, default=4096)
    ap.add_argument("--num_workers", type=int, default=0)
    ap.add_argument("--max_batches", type=int, default=None)
    ap.add_argument("--device", type=str, default=None, choices=[None, "cpu", "cuda"])
    ap.add_argument("--save_json", action="store_true")
    args = ap.parse_args()

    model_pt = os.path.join(args.model_dir, "model.pt")
    cfg_yaml = os.path.join(args.model_dir, "config.yaml")
    if not os.path.exists(model_pt):
        raise FileNotFoundError(f"model.pt not found: {model_pt}")
    if not os.path.exists(args.gold_path):
        raise FileNotFoundError(f"gold_eval.txt not found: {args.gold_path}")

    cfg = load_cfg(cfg_yaml)
    if cfg is None:
        raise RuntimeError("config.yaml not found in model_dir; please keep config.yaml with model.pt")

    # device
    if args.device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    else:
        device = args.device
        if device == "cuda" and not torch.cuda.is_available():
            print("⚠️ cuda requested but not available; fallback to cpu")
            device = "cpu"

    print("device:", device)
    if device == "cuda":
        print("gpu:", torch.cuda.get_device_name(0))

    # preprocessing contract (must match training)
    hash_bins = int(cfg["preprocess"]["hash_bins"])
    seed = int(cfg["split"]["seed"])
    dense_transform = str(cfg["preprocess"]["dense_transform"])
    hp = DACHParams(hash_bins=hash_bins, seed=seed, dense_transform=dense_transform)
    total_embeddings = 26 * hash_bins

    # model params
    embed_dim = int(cfg.get("model", {}).get("embed_dim", 16))
    hidden_dims = tuple(cfg.get("model", {}).get("hidden_dims", [256, 128, 64]))
    dropout = float(cfg.get("model", {}).get("dropout", 0.0))

    # DCNv2 params
    cross_layers = int(cfg.get("dcnv2", {}).get("cross_layers", 3))
    rank = int(cfg.get("dcnv2", {}).get("rank", 32))
    experts = int(cfg.get("dcnv2", {}).get("experts", 4))

    # gold loader
    gold_ds = CriteoDACIterable(args.gold_path, hparams=hp, mode="test")
    gold_loader = DataLoader(
        gold_ds,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        pin_memory=(device == "cuda"),
    )

    # build model + load weights
    model = DCNV2CTR(
        num_embeddings=total_embeddings,
        embed_dim=embed_dim,
        deep_hidden_dims=hidden_dims,
        dropout=dropout,
        cross_layers=cross_layers,
        cross_rank=rank,
        cross_experts=experts,
    ).to(device)

    state = torch.load(model_pt, map_location=device)
    model.load_state_dict(state, strict=True)

    print("model_dir:", args.model_dir)
    print("gold_path:", args.gold_path)
    print("hash_bins:", hash_bins, "total_embeddings:", total_embeddings)
    print("embed_dim:", embed_dim, "deep:", hidden_dims)
    print("dcnv2: cross_layers:", cross_layers, "rank:", rank, "experts:", experts)

    metrics = eval_model(model, gold_loader, device=device, max_batches=args.max_batches)
    print("\n===== GOLD EVAL METRICS (DCN v2) =====")
    for k, v in metrics.items():
        print(f"{k}: {v}")

    if args.save_json:
        out_path = os.path.join(args.model_dir, "eval_gold.json")
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(metrics, f, indent=2)
        print("\n✅ saved:", out_path)


if __name__ == "__main__":
    main()
