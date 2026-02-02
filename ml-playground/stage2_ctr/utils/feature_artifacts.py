import yaml
from typing import Dict, Any


def build_feature_artifacts(
    hash_bins: int,
    seed: int,
    dense_transform: str,
    num_dense: int = 13,
    num_sparse: int = 26,
) -> Dict[str, Any]:
    """
    Build feature artifacts that fully describe feature preprocessing
    for CTR models (DNN / DCN / DCNv2).

    This is the CTR equivalent of:
      - scaler
      - label encoder
      - feature index
    in classic tabular ML.
    """

    artifacts = {
        "schema": {
            "label": "label",
            "dense": [f"I{i}" for i in range(1, num_dense + 1)],
            "sparse": [f"C{i}" for i in range(1, num_sparse + 1)],
            "order": (
                ["label"]
                + [f"I{i}" for i in range(1, num_dense + 1)]
                + [f"C{i}" for i in range(1, num_sparse + 1)]
            ),
        },
        "dense_transform": {
            "type": dense_transform,   # e.g. log1p
        },
        "hashing": {
            "method": "fnv1a_32",
            "hash_bins": hash_bins,
            "seed": seed,
            "num_sparse_fields": num_sparse,
        },
        "field_offsets": {
            f"C{i+1}": i * hash_bins for i in range(num_sparse)
        },
        "total_embeddings": num_sparse * hash_bins,
        "version": "v1",
    }

    return artifacts


def save_feature_artifacts(artifacts: Dict[str, Any], path: str):
    with open(path, "w", encoding="utf-8") as f:
        yaml.safe_dump(artifacts, f, sort_keys=False)

    print("✅ feature_artifacts saved to:", path)


def load_feature_artifacts(path: str) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)
