import os
import sys
import time
import json
import math
import argparse

import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.utils.data import DataLoader

# allow: python .\scripts\train_dcn_v1.py
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from models.dcn_v1_ctr import DCNV1CTR
from datasets.criteo_dac import DACHParams, CriteoDACIterable, make_dataloaders
from utils.feature_artifacts import build_feature_artifacts, save_feature_artifacts

try:
    import yaml
except ImportError:
    yaml = None


@torch.no_grad()
def eval_logloss(model, loader, device, max_batches=None):
    model.eval()
    total = 0.0
    cnt = 0
    for i, (dense, sparse, y) in enumerate(loader):
        if max_batches is not None and i >= max_batches:
            break
        dense = dense.to(device, non_blocking=True)
        sparse = sparse.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)

        logit = model(dense, sparse)
        loss = nn.functional.binary_cross_entropy_with_logits(logit, y, reduction="sum")
        total += loss.item()
        cnt += y.numel()
    return total / max(1, cnt)


def load_cfg(path="configs/local.yaml"):
    if yaml is None:
        raise RuntimeError("PyYAML not installed. Run: pip install pyyaml")
    return yaml.safe_load(open(path, "r", encoding="utf-8"))


def save_run(out_root, run_name, model, cfg, metrics, feature_artifacts):
    os.makedirs(out_root, exist_ok=True)
    run_dir = os.path.join(out_root, run_name)
    os.makedirs(run_dir, exist_ok=True)

    torch.save(model.state_dict(), os.path.join(run_dir, "model.pt"))
    with open(os.path.join(run_dir, "config.yaml"), "w", encoding="utf-8") as f:
        yaml.safe_dump(cfg, f, sort_keys=False)
    with open(os.path.join(run_dir, "metrics.json"), "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)

    save_feature_artifacts(feature_artifacts, os.path.join(run_dir, "feature_artifacts.yaml"))
    print("✅ saved:", run_dir)
    return run_dir


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", type=str, default="configs/local.yaml")
    ap.add_argument("--out_root", type=str, default=r"D:\repo\large data\model")
    ap.add_argument("--run_name", type=str, default=None)
    ap.add_argument("--device", type=str, default=None, choices=[None, "cpu", "cuda"])
    args = ap.parse_args()

    cfg = load_cfg(args.config)

    # device
    if args.device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    else:
        device = args.device
        if device == "cuda" and not torch.cuda.is_available():
            device = "cpu"

    print("device:", device)
    if device == "cuda":
        print("gpu:", torch.cuda.get_device_name(0))

    # paths
    dac_dir = cfg["data"]["dac_dir"]
    train_path = os.path.normpath(os.path.join(dac_dir, cfg["data"]["train_file"]))
    gold_path = os.path.normpath(os.path.join(dac_dir, "_gold", "gold_eval.txt"))

    # hparams / preprocessing (SAME as DNN)
    hash_bins = int(cfg["preprocess"]["hash_bins"])
    seed = int(cfg["split"]["seed"])
    dense_transform = str(cfg["preprocess"]["dense_transform"])
    hp = DACHParams(hash_bins=hash_bins, seed=seed, dense_transform=dense_transform)

    # loaders
    train_loader, _, total_embeddings = make_dataloaders(
        train_path=train_path,
        hp=hp,
        val_ratio=float(cfg["split"]["val_ratio"]),
        batch_size=int(cfg["loader"]["batch_size"]),
        num_workers=int(cfg["loader"]["num_workers"]),
        pin_memory=bool(cfg["loader"]["pin_memory"]),
        prefetch_factor=int(cfg["loader"].get("prefetch_factor", 2)),
    )

    # gold eval loader (fixed dataset)
    gold_ds = CriteoDACIterable(gold_path, hparams=hp, mode="test")
    gold_loader = DataLoader(
        gold_ds,
        batch_size=int(cfg["loader"]["batch_size"]),
        num_workers=int(cfg["loader"]["num_workers"]),
        pin_memory=(device == "cuda"),
    )

    # model
    embed_dim = int(cfg.get("model", {}).get("embed_dim", 16))
    hidden_dims = tuple(cfg.get("model", {}).get("hidden_dims", [256, 128, 64]))
    dropout = float(cfg.get("model", {}).get("dropout", 0.0))

    cross_layers = int(cfg.get("dcn", {}).get("cross_layers", 3))

    model = DCNV1CTR(
        num_embeddings=total_embeddings,
        embed_dim=embed_dim,
        deep_hidden_dims=hidden_dims,
        dropout=dropout,
        cross_layers=cross_layers,
    ).to(device)

    # (optional but recommended) init output bias to dataset prior
    # Use your measured p=0.243425 or compute it separately
    p = float(cfg.get("label", {}).get("p_prior", 0.243425))
    with torch.no_grad():
        model.fc_out.bias.fill_(math.log(p / (1 - p)))

    optimizer = AdamW(model.parameters(), lr=float(cfg["train"]["lr"]), weight_decay=1e-6)
    criterion = nn.BCEWithLogitsLoss()

    steps_per_epoch = int(cfg["train"]["steps_per_epoch"])
    epochs = int(cfg["train"]["epochs"])
    log_every = int(cfg["train"]["log_every"])

    print("train_path:", train_path)
    print("gold_path:", gold_path)
    print("hash_bins:", hash_bins, "total_embeddings:", total_embeddings)
    print("embed_dim:", embed_dim, "cross_layers:", cross_layers, "deep:", hidden_dims)

    # artifacts (SAME contract for DNN/DCN)
    feature_artifacts = build_feature_artifacts(
        hash_bins=hash_bins,
        seed=seed,
        dense_transform=dense_transform,
    )

    # train
    global_step = 0
    model.train()
    for ep in range(1, epochs + 1):
        t0 = time.time()
        running = 0.0

        for step, (dense, sparse, y) in enumerate(train_loader, start=1):
            dense = dense.to(device, non_blocking=True)
            sparse = sparse.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)

            logit = model(dense, sparse)
            loss = criterion(logit, y)

            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()

            running += loss.item()
            global_step += 1

            if global_step % log_every == 0:
                print(f"ep={ep} step={global_step} train_loss={running/log_every:.4f}")
                running = 0.0

            if step >= steps_per_epoch:
                break

        gold_logloss = eval_logloss(model, gold_loader, device=device, max_batches=500)
        print(f"[epoch {ep}] gold_logloss={gold_logloss:.4f} epoch_time={time.time()-t0:.1f}s")

    # save
    if args.run_name is None:
        run_name = time.strftime("dcn_v1_%Y%m%d_%H%M%S")
    else:
        run_name = args.run_name

    metrics = {
        "gold_logloss": float(gold_logloss),
        "hash_bins": hash_bins,
        "total_embeddings": int(total_embeddings),
        "embed_dim": embed_dim,
        "cross_layers": cross_layers,
        "deep_hidden_dims": list(hidden_dims),
        "p_prior": p,
    }

    save_run(args.out_root, run_name, model, cfg, metrics, feature_artifacts)


if __name__ == "__main__":
    main()
