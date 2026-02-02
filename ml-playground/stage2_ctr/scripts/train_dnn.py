import os
import time
import yaml
import torch
import torch.nn as nn
from torch.optim import AdamW

from datasets.criteo_dac import DACHParams, make_dataloaders, CriteoDACIterable 
from models.dnn_ctr import DNNCTR
from torch.utils.data import DataLoader

import json
from datetime import datetime

from utils.feature_artifacts import save_feature_artifacts, build_feature_artifacts

import math


@torch.no_grad()
def eval_logloss(model, loader, device, max_batches=200):
    model.eval()
    bce = nn.BCEWithLogitsLoss(reduction="sum")
    total_loss = 0.0
    total_cnt = 0

    for i, (dense, sparse, y) in enumerate(loader):
        if max_batches is not None and i >= max_batches:
            break
        dense = dense.to(device)
        sparse = sparse.to(device)
        y = y.to(device)

        logit = model(dense, sparse)
        loss = bce(logit, y)
        total_loss += loss.item()
        total_cnt += y.numel()

    return total_loss / max(1, total_cnt)

def save_run(model, cfg, metrics: dict, out_root=r"D:\repo\large data\model"):
    os.makedirs(out_root, exist_ok=True)
    run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = os.path.join(out_root, f"dnn_baseline_{run_id}")
    os.makedirs(run_dir, exist_ok=True)

    # 1) model weights
    torch.save(model.state_dict(), os.path.join(run_dir, "model.pt"))

    # 2) config snapshot
    with open(os.path.join(run_dir, "config.yaml"), "w", encoding="utf-8") as f:
        yaml.safe_dump(cfg, f, sort_keys=False)

    # 3) metrics
    with open(os.path.join(run_dir, "metrics.json"), "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)
    artifacts_path = os.path.join(run_dir, "feature_artifacts.yaml")
    artifacts = build_feature_artifacts(
        hash_bins=int(cfg["preprocess"]["hash_bins"]),
        seed=int(cfg["split"]["seed"]),
        dense_transform=str(cfg["preprocess"]["dense_transform"]),
    )
    save_feature_artifacts(artifacts, artifacts_path)
    print("✅ saved run to:", run_dir)

def main():
    cfg = yaml.safe_load(open("configs/local.yaml", "r", encoding="utf-8"))

    dac_dir = cfg["data"]["dac_dir"]
    train_file = cfg["data"]["train_file"]
    train_path = os.path.normpath(os.path.join(dac_dir, train_file))
    gold_path = os.path.normpath(os.path.join(dac_dir, "_gold", "gold_eval.txt"))
    hp = DACHParams(
        hash_bins=int(cfg["preprocess"]["hash_bins"]),
        seed=int(cfg["split"]["seed"]),
        dense_transform=str(cfg["preprocess"]["dense_transform"]),
    )

    train_loader, val_loader, total_embeddings = make_dataloaders(
        train_path=train_path,
        hp=hp if "hp" in make_dataloaders.__code__.co_varnames else hp,  # 兼容你本地版本
        val_ratio=float(cfg["split"]["val_ratio"]),
        batch_size=int(cfg["loader"]["batch_size"]),
        num_workers=int(cfg["loader"]["num_workers"]),
        pin_memory=bool(cfg["loader"]["pin_memory"]),
        prefetch_factor=int(cfg["loader"]["prefetch_factor"]),
    )

    gold_ds = CriteoDACIterable(gold_path, hparams=hp, mode="test")
    gold_loader = DataLoader(
        gold_ds,
        batch_size=int(cfg["loader"]["batch_size"]),
        num_workers=int(cfg["loader"]["num_workers"]),
        pin_memory=bool(cfg["loader"]["pin_memory"]),
    )
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("device:", device)
    print("train_path:", train_path)
    print("total_embeddings:", total_embeddings)

    # ===== model =====
    model = DNNCTR(
        num_embeddings=total_embeddings,
        embed_dim=int(cfg.get("model", {}).get("embed_dim", 16)),
        hidden_dims=tuple(cfg.get("model", {}).get("hidden_dims", [256, 128, 64])),
        dropout=float(cfg.get("model", {}).get("dropout", 0.0)),
    ).to(device)

    # 初始化最后一层偏置，使得初始预测与正样本比例一致
    p = 0.243425 # 你可以改成你的数据集的正样本比例
    model.mlp[-1].bias.data.fill_(math.log(p / (1 - p)))

    print("torch.cuda.is_available:", torch.cuda.is_available())
    if torch.cuda.is_available():
        print("gpu:", torch.cuda.get_device_name(0))

    print("model device:", next(model.parameters()).device)

    optimizer = AdamW(model.parameters(), lr=float(cfg.get("train", {}).get("lr", 1e-3)), weight_decay=1e-6)
    criterion = nn.BCEWithLogitsLoss()

    # ===== train loop =====
    steps_per_epoch = int(cfg.get("train", {}).get("steps_per_epoch", 2000))
    epochs = int(cfg.get("train", {}).get("epochs", 2))
    log_every = int(cfg.get("train", {}).get("log_every", 100))

    model.train()
    global_step = 0
    for ep in range(1, epochs + 1):
        t0 = time.time()
        running = 0.0
        for step, (dense, sparse, y) in enumerate(train_loader, start=1):
            dense = dense.to(device, non_blocking=True)
            sparse = sparse.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)
            if global_step % 100 == 0:
                print("batch_pos_rate:", float(y.mean().item()))
            if global_step == 1:
                print("dense device:", dense.device, "sparse device:", sparse.device, "y device:", y.device)
            
            logit = model(dense, sparse)
            loss = criterion(logit, y)

            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()

            running += loss.item()
            global_step += 1

            if global_step % log_every == 0:
                avg = running / log_every
                running = 0.0
                print(f"ep={ep} step={global_step} train_loss={avg:.4f}")

            if step >= steps_per_epoch:
                break

        val_logloss = eval_logloss(model, val_loader, device=device, max_batches=200)
        gold_logloss = eval_logloss(model, gold_loader, device=device, max_batches=200)
        print(f"[epoch {ep}] val_logloss={val_logloss:.4f} epoch_time={time.time()-t0:.1f}s")
        print(f"[epoch {ep}] gold_logloss={gold_loader:.4f}")
        metrics = {
            "val_logloss": float(val_logloss),
            "gold_logloss": float(gold_logloss),
            "total_embeddings": int(total_embeddings),
            "hash_bins": int(cfg["preprocess"]["hash_bins"]),
            "embed_dim": int(cfg.get("model", {}).get("embed_dim", 16)),
        }
        save_run(model, cfg, metrics)
        
    print("✅ done.")


if __name__ == "__main__":
    main()
