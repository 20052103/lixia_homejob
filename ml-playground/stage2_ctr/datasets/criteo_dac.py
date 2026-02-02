import os
import math
import random
from dataclasses import dataclass
from typing import Iterator, List, Optional, Tuple

import torch
from torch.utils.data import IterableDataset, DataLoader


NUM_DENSE = 13
NUM_SPARSE = 26


def _detect_delim(line: str) -> str:
    # Criteo 常见是 tab 分隔；有些版本是空格
    return "\t" if ("\t" in line) else " "


def _safe_int(x: str) -> int:
    if x == "" or x is None:
        return 0
    try:
        return int(x)
    except ValueError:
        # 有些数据里可能出现非整数字符，保底
        return 0


def _hash_cat(val: str, seed: int = 0) -> int:
    # Python 内置 hash 每次进程可能变（受 PYTHONHASHSEED 影响）
    # 用一个稳定的简单 hash：FNV-1a 风格（够用且快）
    if val is None or val == "":
        return 0
    h = 2166136261 ^ seed
    for ch in val:
        h ^= ord(ch)
        h = (h * 16777619) & 0xFFFFFFFF
    return h


@dataclass
class DACHParams:
    hash_bins: int = 2_000_000
    seed: int = 42
    dense_transform: str = "log1p"  # "none" | "log1p"


class CriteoDACIterable(IterableDataset):
    """
    流式读取 Criteo DAC：
    - 输出:
      dense: FloatTensor [13]
      sparse: LongTensor [26]   (每个field hash到 [0, hash_bins) 后再加 offset 变成全局 id)
      label:  FloatTensor []    (0/1)
    """

    def __init__(
        self,
        path: str,
        hparams: DACHParams,
        mode: str = "train",
        val_ratio: float = 0.05,
    ):
        super().__init__()
        self.path = path
        self.hp = hparams
        assert mode in ("train", "val", "test")
        self.mode = mode
        self.val_ratio = float(val_ratio)

        # 每个 field 一个 offset，保证 embedding index 全局唯一
        # global_id = field_id * hash_bins + hashed_id
        self.field_offsets = torch.arange(NUM_SPARSE, dtype=torch.long) * self.hp.hash_bins
        self.total_embeddings = NUM_SPARSE * self.hp.hash_bins

    def _dense_transform(self, x: int) -> float:
        # Criteo dense 常是非负整数；缺失记为 0
        if self.hp.dense_transform == "log1p":
            return float(math.log1p(max(x, 0)))
        return float(x)

    def _is_val_line(self, line_idx: int) -> bool:
        # 可复现 split：用 line_idx 做一个稳定“伪随机”
        # val_ratio=0.05 => 大约 1/20 行进 val
        # 这里用线性同余生成器映射到 [0,1)
        a = 1103515245
        c = 12345
        m = 2**31
        r = (a * (line_idx + self.hp.seed) + c) % m
        return (r / m) < self.val_ratio

    def __iter__(self) -> Iterator[Tuple[torch.Tensor, torch.Tensor, torch.Tensor]]:
        worker_info = torch.utils.data.get_worker_info()
        worker_id = worker_info.id if worker_info else 0
        num_workers = worker_info.num_workers if worker_info else 1

        with open(self.path, "r", encoding="utf-8", errors="ignore") as f:
            first = f.readline()
            if not first:
                return
            delim = _detect_delim(first)
            # rewind: 处理第一行
            f.seek(0)

            for line_idx, line in enumerate(f):
                # 多 worker 分片：每个 worker 处理自己负责的行
                if (line_idx % num_workers) != worker_id:
                    continue

                line = line.rstrip("\n")
                if not line:
                    continue

                cols = line.split(delim)
                # 允许某些版本列数不足/多余，尽量容错
                # 标准：1(label) + 13(dense) + 26(sparse) = 40
                if len(cols) < 1 + NUM_DENSE:
                    continue

                # split
                if self.mode in ("train", "val"):
                    is_val = self._is_val_line(line_idx)
                    if self.mode == "train" and is_val:
                        continue
                    if self.mode == "val" and (not is_val):
                        continue

                label = _safe_int(cols[0])
                dense_raw = cols[1 : 1 + NUM_DENSE]
                sparse_raw = cols[1 + NUM_DENSE : 1 + NUM_DENSE + NUM_SPARSE]

                dense = torch.tensor([self._dense_transform(_safe_int(x)) for x in dense_raw], dtype=torch.float32)

                # sparse hashing + per-field offset
                sparse_ids: List[int] = []
                # 如果 sparse 列不够，用空补齐
                if len(sparse_raw) < NUM_SPARSE:
                    sparse_raw = sparse_raw + [""] * (NUM_SPARSE - len(sparse_raw))

                for j, v in enumerate(sparse_raw[:NUM_SPARSE]):
                    hid = _hash_cat(v, seed=self.hp.seed + j) % self.hp.hash_bins
                    gid = int(self.field_offsets[j].item() + hid)
                    sparse_ids.append(gid)

                sparse = torch.tensor(sparse_ids, dtype=torch.long)
                y = torch.tensor(float(1 if label > 0 else 0), dtype=torch.float32)

                yield dense, sparse, y


def make_dataloaders(
    train_path: str,
    hp: DACHParams,
    val_ratio: float,
    batch_size: int,
    num_workers: int = 4,
    pin_memory: bool = True,
    prefetch_factor: int = 2,
):
    train_ds = CriteoDACIterable(train_path, hparams=hp, mode="train", val_ratio=val_ratio)
    val_ds = CriteoDACIterable(train_path, hparams=hp, mode="val", val_ratio=val_ratio)


    def _collate(batch):
        dense = torch.stack([b[0] for b in batch], dim=0)   # [B, 13]
        sparse = torch.stack([b[1] for b in batch], dim=0)  # [B, 26]
        y = torch.stack([b[2] for b in batch], dim=0)       # [B]
        return dense, sparse, y

    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        num_workers=num_workers,
        pin_memory=pin_memory,
        prefetch_factor=prefetch_factor if num_workers > 0 else None,
        collate_fn=_collate,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=batch_size,
        num_workers=num_workers,
        pin_memory=pin_memory,
        prefetch_factor=prefetch_factor if num_workers > 0 else None,
        collate_fn=_collate,
    )
    return train_loader, val_loader, train_ds.total_embeddings
