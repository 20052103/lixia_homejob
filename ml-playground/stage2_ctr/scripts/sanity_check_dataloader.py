import os
import yaml
import torch

from datasets.criteo_dac import DACHParams, make_dataloaders


def main():
    cfg = yaml.safe_load(open("configs/local.yaml", "r", encoding="utf-8"))
    dac_dir = cfg["data"]["dac_dir"]
    train_file = cfg["data"]["train_file"]
    train_path = os.path.join(dac_dir, train_file)

    hp = DACHParams(
        hash_bins=int(cfg["preprocess"]["hash_bins"]),
        seed=int(cfg["split"]["seed"]),
        dense_transform=str(cfg["preprocess"]["dense_transform"]),
    )

    train_loader, val_loader, total_embeddings = make_dataloaders(
        train_path=train_path,
        hp=hp,
        val_ratio=float(cfg["split"]["val_ratio"]),
        batch_size=int(cfg["loader"]["batch_size"]),
        num_workers=int(cfg["loader"]["num_workers"]),
        pin_memory=bool(cfg["loader"]["pin_memory"]),
        prefetch_factor=int(cfg["loader"]["prefetch_factor"]),
    )

    print("train_path:", train_path)
    print("total_embeddings:", total_embeddings)

    dense, sparse, y = next(iter(train_loader))
    print("dense:", dense.shape, dense.dtype, "min/max:", dense.min().item(), dense.max().item())
    print("sparse:", sparse.shape, sparse.dtype, "min/max:", sparse.min().item(), sparse.max().item())
    print("y:", y.shape, y.dtype, "pos_rate(batch):", y.mean().item())

    # 简单检查：sparse 是否都在合法范围
    assert sparse.min().item() >= 0
    assert sparse.max().item() < total_embeddings
    print("✅ dataloader sanity check passed.")


if __name__ == "__main__":
    main()
