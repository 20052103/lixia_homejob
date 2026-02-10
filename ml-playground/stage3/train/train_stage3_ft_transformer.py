from datetime import datetime
import os
import sys
import time
from typing import List, Tuple, Optional

import torch
import torch.nn as nn
import yaml
import json

# Add repo root to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from stage3.models.ft_transformer import FTTransformerCTR
from stage2_ctr.utils.feature_artifacts import save_feature_artifacts, build_feature_artifacts
from torch.utils.data import DataLoader
from stage2_ctr.datasets.criteo_dac import DACHParams, CriteoDACIterable, make_dataloaders


@torch.no_grad()
def eval_logloss(model, loader, device, max_batches=200, hash_bins=None):
    model.eval()
    bce = nn.BCEWithLogitsLoss(reduction="sum")
    total_loss = 0.0
    total_cnt = 0

    for i, (dense, sparse, y) in enumerate(loader):
        if max_batches is not None and i >= max_batches:
            break

        dense = dense.to(device, non_blocking=True)
        sparse = sparse.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True).float().view(-1)

        if hash_bins is not None:
            sparse = (sparse % hash_bins).long()
        else:
            sparse = sparse.long()

        logit = model(dense, sparse).view(-1)
        loss = bce(logit, y)
        total_loss += loss.item()
        total_cnt += y.numel()

    return total_loss / max(1, total_cnt)


def save_run(model, cfg, metrics: dict, out_root=r"D:\repo\large data\model"):
    os.makedirs(out_root, exist_ok=True)
    run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = os.path.join(out_root, f"dnn_ft_transformer_{run_id}")
    os.makedirs(run_dir, exist_ok=True)

    torch.save(model.state_dict(), os.path.join(run_dir, "model.pt"))

    with open(os.path.join(run_dir, "config.yaml"), "w", encoding="utf-8") as f:
        yaml.safe_dump(cfg, f, sort_keys=False)

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


def build_optimizers_embedding_sgd_other_adamw(
    model: torch.nn.Module,
    lr_embed: float,
    lr_other: float,
    other_weight_decay: float = 0.01,
    embed_momentum: float = 0.0,
) -> Tuple[Optional[torch.optim.Optimizer], Optional[torch.optim.Optimizer]]:
    """
    Embedding params -> SGD (memory friendly)
    Other params     -> AdamW
    """
    embed_params: List[torch.nn.Parameter] = []
    other_params: List[torch.nn.Parameter] = []

    for name, p in model.named_parameters():
        if not p.requires_grad:
            continue
        lname = name.lower()
        # FT tokenizer uses "cat_emb"
        if ("cat_emb" in lname) or ("embedding" in lname) or (".emb" in lname):
            embed_params.append(p)
        else:
            other_params.append(p)

    optim_embed = None
    if len(embed_params) > 0:
        optim_embed = torch.optim.SGD(
            embed_params,
            lr=lr_embed,
            momentum=embed_momentum,
            weight_decay=0.0,
        )

    optim_other = None
    if len(other_params) > 0:
        optim_other = torch.optim.AdamW(
            other_params,
            lr=lr_other,
            weight_decay=other_weight_decay,
        )

    n_embed = sum(p.numel() for p in embed_params)
    n_other = sum(p.numel() for p in other_params)
    print(f"[optim] embed tensors={len(embed_params)} params={n_embed:,} -> SGD(lr={lr_embed})")
    print(f"[optim] other tensors={len(other_params)} params={n_other:,} -> AdamW(lr={lr_other}, wd={other_weight_decay})")
    return optim_embed, optim_other


def main():
    # Load config
    script_dir = os.path.dirname(os.path.abspath(__file__))
    config_path = os.path.join(script_dir, "..", "configs", "local.yaml")
    cfg = yaml.safe_load(open(config_path, "r", encoding="utf-8"))

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("torch:", torch.__version__)
    print("cuda available:", torch.cuda.is_available())
    print("cuda version:", torch.version.cuda)
    print("selected device:", device)

    # Train config
    epochs = int(cfg["train"]["epochs"])
    steps_per_epoch = int(cfg["train"]["steps_per_epoch"])
    log_every = int(cfg["train"]["log_every"])
    lr_embed = float(cfg["train"]["lr_embed"])
    lr_other = float(cfg["train"]["lr_other"])
    wd_other = float(cfg["train"]["wd_other"])
    embed_momentum = float(cfg["train"]["embed_momentum"])

    # Model config (reuse same keys; add optional ffn_mult, pooling)
    d_token = int(cfg["model"]["d_token"])
    n_heads = int(cfg["model"]["n_heads"])
    n_layers = int(cfg["model"]["n_layers"])
    dropout = float(cfg["model"]["dropout"])
    dense_token_mode = str(cfg["model"]["dense_token_mode"])
    pooling = str(cfg["model"].get("pooling", "cls"))  # FT default is cls
    ffn_mult = int(cfg["model"].get("ffn_mult", 4))
    ckpt_dir = str(cfg["model"]["ckpt_dir"])

    # Data paths
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
        hp=hp if "hp" in make_dataloaders.__code__.co_varnames else hp,
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

    # Model
    num_dense = 13
    num_cat = 26
    hash_bins = int(cfg["preprocess"]["hash_bins"])

    model = FTTransformerCTR(
        num_dense=num_dense,
        num_cat=num_cat,
        hash_bins=hash_bins,
        d_token=d_token,
        n_heads=n_heads,
        n_layers=n_layers,
        ffn_mult=ffn_mult,
        dropout=dropout,
        dense_token_mode=dense_token_mode,
        pooling=pooling,
    ).to(device)

    criterion = nn.BCEWithLogitsLoss()

    optim_embed, optim_other = build_optimizers_embedding_sgd_other_adamw(
        model,
        lr_embed=lr_embed,
        lr_other=lr_other,
        other_weight_decay=wd_other,
        embed_momentum=embed_momentum,
    )

    os.makedirs(ckpt_dir, exist_ok=True)
    global_step = 1

    for ep in range(1, epochs + 1):
        t0 = time.time()
        running = 0.0
        model.train()

        for step, (dense, sparse, y) in enumerate(train_loader, start=1):
            dense = dense.to(device, non_blocking=True)
            sparse = sparse.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True).float().view(-1)

            # dataset emits global IDs (field_offset + local_id) -> convert to local_id
            sparse = (sparse % hash_bins).long()

            if global_step % 100 == 0:
                print("batch_pos_rate:", float(y.mean().item()))
            if global_step == 1:
                print("dense device:", dense.device, "sparse device:", sparse.device, "y device:", y.device)
                print("dense shape:", tuple(dense.shape), "sparse shape:", tuple(sparse.shape), "y shape:", tuple(y.shape))

            logit = model(dense, sparse).view(-1)
            loss = criterion(logit, y)

            if optim_embed is not None:
                optim_embed.zero_grad(set_to_none=True)
            if optim_other is not None:
                optim_other.zero_grad(set_to_none=True)

            loss.backward()

            if optim_embed is not None:
                optim_embed.step()
            if optim_other is not None:
                optim_other.step()

            running += float(loss.item())
            global_step += 1

            if global_step % log_every == 0:
                avg = running / log_every
                running = 0.0
                print(f"ep={ep} step={global_step} train_loss={avg:.4f}")

            if step >= steps_per_epoch:
                break

        # Save checkpoint
        ckpt_path = os.path.join(ckpt_dir, f"ft_transformer_ep{ep}.pt")
        torch.save({"model": model.state_dict(), "cfg": cfg}, ckpt_path)
        print(f"[epoch {ep}] saved={ckpt_path} epoch_time={time.time()-t0:.1f}s")

        # Eval
        val_logloss = eval_logloss(model, val_loader, device=device, max_batches=200, hash_bins=hash_bins)
        gold_logloss = eval_logloss(model, gold_loader, device=device, max_batches=None, hash_bins=hash_bins)
        print(f"[epoch {ep}] val_logloss={val_logloss:.4f}")
        print(f"[epoch {ep}] gold_logloss={gold_logloss:.4f}")

        metrics = {
            "model": "ft_transformer",
            "val_logloss": float(val_logloss),
            "gold_logloss": float(gold_logloss),
            "total_embeddings": int(total_embeddings),
            "hash_bins": int(hash_bins),
            "d_token": int(d_token),
            "n_heads": int(n_heads),
            "n_layers": int(n_layers),
            "ffn_mult": int(ffn_mult),
            "pooling": str(pooling),
        }
        save_run(model, cfg, metrics)

    print("✅ done.")


if __name__ == "__main__":
    main()
