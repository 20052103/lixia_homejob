import os
import sys
import time
import json
import argparse

import numpy as np

# ---- make imports work even with: python .\scripts\train_gbdt_lgbm.py
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from datasets.criteo_dac import CriteoDACIterable, DACHParams

import lightgbm as lgb
from sklearn.metrics import log_loss, roc_auc_score


def dataset_to_numpy(ds, max_rows=None):
    """
    Convert (dense, sparse, y) stream into numpy arrays for LightGBM.

    dense: torch.Tensor [13]
    sparse: torch.Tensor [26] (hashed int ids)
    y: torch.Tensor scalar float (0/1)
    """
    X = []
    y = []
    for i, (dense, sparse, label) in enumerate(ds):
        if max_rows is not None and i >= max_rows:
            break

        # IMPORTANT: keep same feature order: [dense..., sparse...]
        feat = np.concatenate([dense.numpy(), sparse.numpy()]).astype(np.float32)
        X.append(feat)
        y.append(float(label.item()))

        if (i + 1) % 200_000 == 0:
            print(f"loaded {i+1} rows...")

    X = np.stack(X, axis=0)
    y = np.array(y, dtype=np.float32)
    return X, y


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--train_path", type=str, default=r"D:\repo\large data\criteo\dac\train.txt")
    ap.add_argument("--gold_path", type=str, default=r"D:\repo\large data\criteo\dac\_gold\gold_eval.txt")
    ap.add_argument("--out_root", type=str, default=r"D:\repo\large data\model")

    # keep preprocessing identical to DNN/DCN
    ap.add_argument("--hash_bins", type=int, default=2_000_000)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--dense_transform", type=str, default="log1p")

    # data size controls (GBDT does not need the full 5GB to be a useful low-bound)
    ap.add_argument("--max_train_rows", type=int, default=2_000_000)
    ap.add_argument("--max_gold_rows", type=int, default=None)

    # LGBM params
    ap.add_argument("--num_boost_round", type=int, default=500)
    ap.add_argument("--early_stopping_rounds", type=int, default=30)
    ap.add_argument("--learning_rate", type=float, default=0.05)
    ap.add_argument("--num_leaves", type=int, default=64)
    ap.add_argument("--min_data_in_leaf", type=int, default=100)
    ap.add_argument("--feature_fraction", type=float, default=0.8)
    ap.add_argument("--bagging_fraction", type=float, default=0.8)
    ap.add_argument("--bagging_freq", type=int, default=5)

    args = ap.parse_args()

    if not os.path.exists(args.train_path):
        raise FileNotFoundError(f"train_path not found: {args.train_path}")
    if not os.path.exists(args.gold_path):
        raise FileNotFoundError(f"gold_path not found: {args.gold_path}")

    # Build preprocessing hparams (same contract as deep models)
    hp = DACHParams(hash_bins=args.hash_bins, seed=args.seed, dense_transform=args.dense_transform)

    print("train_path:", args.train_path)
    print("gold_path:", args.gold_path)
    print("hash_bins:", args.hash_bins, "seed:", args.seed, "dense_transform:", args.dense_transform)
    print("max_train_rows:", args.max_train_rows, "max_gold_rows:", args.max_gold_rows)

    # Load data via your existing dataset pipeline
    print("\nLoading train dataset via CriteoDACIterable...")
    train_ds = CriteoDACIterable(args.train_path, hparams=hp, mode="train")
    X_train, y_train = dataset_to_numpy(train_ds, max_rows=args.max_train_rows)

    print("\nLoading gold dataset via CriteoDACIterable...")
    gold_ds = CriteoDACIterable(args.gold_path, hparams=hp, mode="test")
    X_gold, y_gold = dataset_to_numpy(gold_ds, max_rows=args.max_gold_rows)

    print("\nShapes:")
    print("X_train:", X_train.shape, "y_train:", y_train.shape, "pos_rate:", float(y_train.mean()))
    print("X_gold :", X_gold.shape,  "y_gold :", y_gold.shape,  "pos_rate:", float(y_gold.mean()))

    # Mark sparse columns as categorical features for LightGBM
    num_dense = 13
    num_sparse = 26
    categorical_features = list(range(num_dense, num_dense + num_sparse))

    train_set = lgb.Dataset(
        X_train,
        label=y_train,
        categorical_feature=categorical_features,
        free_raw_data=False,
    )
    gold_set = lgb.Dataset(
        X_gold,
        label=y_gold,
        categorical_feature=categorical_features,
        reference=train_set,
        free_raw_data=False,
    )

    params = {
        "objective": "binary",
        "metric": ["binary_logloss", "auc"],
        "learning_rate": args.learning_rate,
        "num_leaves": args.num_leaves,
        "max_depth": -1,
        "min_data_in_leaf": args.min_data_in_leaf,
        "feature_fraction": args.feature_fraction,
        "bagging_fraction": args.bagging_fraction,
        "bagging_freq": args.bagging_freq,
        "seed": args.seed,
        "verbosity": -1,
    }

    print("\nTraining LightGBM...")
    t0 = time.time()
    booster = lgb.train(
        params,
        train_set,
        num_boost_round=args.num_boost_round,
        valid_sets=[gold_set],
        valid_names=["gold"],
        callbacks=[
            lgb.early_stopping(stopping_rounds=args.early_stopping_rounds, verbose=True),
            lgb.log_evaluation(period=20),
        ],
    )
    print("Training time (s):", round(time.time() - t0, 1))

    # Evaluate on gold
    preds = booster.predict(X_gold, num_iteration=booster.best_iteration)
    gold_logloss = float(log_loss(y_gold, preds, eps=1e-12))
    gold_auc = float(roc_auc_score(y_gold, preds))
    avg_pred = float(np.mean(preds))
    pos_rate = float(np.mean(y_gold))

    print("\n===== GBDT (LightGBM) GOLD EVAL =====")
    print("logloss:", gold_logloss)
    print("auc:", gold_auc)
    print("pos_rate:", pos_rate)
    print("avg_pred:", avg_pred)
    print("best_iteration:", int(booster.best_iteration))

    # Save run
    run_name = time.strftime("gbdt_lgbm_%Y%m%d_%H%M%S")
    run_dir = os.path.join(args.out_root, run_name)
    os.makedirs(run_dir, exist_ok=True)

    booster.save_model(os.path.join(run_dir, "lgbm_model.txt"))

    metrics = {
        "logloss": gold_logloss,
        "auc": gold_auc,
        "pos_rate": pos_rate,
        "avg_pred": avg_pred,
        "num_samples": int(len(y_gold)),
        "best_iteration": int(booster.best_iteration),
    }
    with open(os.path.join(run_dir, "eval_gold.json"), "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)

    run_cfg = {
        "train_path": args.train_path,
        "gold_path": args.gold_path,
        "hash_bins": args.hash_bins,
        "seed": args.seed,
        "dense_transform": args.dense_transform,
        "max_train_rows": args.max_train_rows,
        "max_gold_rows": args.max_gold_rows,
        "lgbm_params": params,
        "num_boost_round": args.num_boost_round,
        "early_stopping_rounds": args.early_stopping_rounds,
    }
    with open(os.path.join(run_dir, "run_config.json"), "w", encoding="utf-8") as f:
        json.dump(run_cfg, f, indent=2)

    print("\n✅ Saved run to:", run_dir)
    print("  - lgbm_model.txt")
    print("  - eval_gold.json")
    print("  - run_config.json")


if __name__ == "__main__":
    main()
