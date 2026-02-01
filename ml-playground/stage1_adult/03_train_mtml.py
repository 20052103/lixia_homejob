import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
from sklearn.metrics import roc_auc_score

class AdultDataset(Dataset):
    def __init__(self, X_num, X_cat, yA, yB):
        self.X_num = torch.from_numpy(X_num)
        self.X_cat = torch.from_numpy(X_cat)
        self.yA = torch.from_numpy(yA).float()
        self.yB = torch.from_numpy(yB).float()

    def __len__(self): return self.yA.shape[0]

    def __getitem__(self, idx):
        return self.X_num[idx], self.X_cat[idx], self.yA[idx], self.yB[idx]

# Multi-Task Multi-Loss (MTML) Model for Adult dataset
# This model performs two related binary classification tasks:
# - Task A: Income prediction (>50K or <=50K)
# - Task B: Another binary target (potentially demographic or secondary classification)
# The model uses shared layers with task-specific output heads
class MTMLModel(nn.Module):
    def __init__(self, num_dim, cat_sizes, emb_dim=16, hidden=(256,128,64), dropout=0.1):
        super().__init__()
        # Embedding layer for categorical features
        self.embs = nn.ModuleList([nn.Embedding(int(sz), emb_dim) for sz in cat_sizes])
        in_dim = num_dim + emb_dim * len(cat_sizes)

        # Shared layers: dense network to learn common representations
        layers = []
        d = in_dim
        for h in hidden:
            layers += [nn.Linear(d, h), nn.ReLU(), nn.Dropout(dropout)]
            d = h
        self.shared = nn.Sequential(*layers)

        # Task-specific output heads for each target
        self.headA = nn.Linear(d, 1)  # Head for Task A (income prediction)
        self.headB = nn.Linear(d, 1)  # Head for Task B (secondary target)

    def forward(self, x_num, x_cat):
        emb_list = [emb(x_cat[:, i]) for i, emb in enumerate(self.embs)]
        x = torch.cat([x_num] + emb_list, dim=1)
        h = self.shared(x)
        return self.headA(h).squeeze(1), self.headB(h).squeeze(1)

def bce_logits(logits, y):
    return nn.functional.binary_cross_entropy_with_logits(logits, y)

@torch.no_grad()
def eval_auc(model, loader, device):
    model.eval()
    allA_p, allA_y = [], []
    allB_p, allB_y = [], []
    for x_num, x_cat, yA, yB in loader:
        x_num = x_num.to(device, non_blocking=True)
        x_cat = x_cat.to(device, non_blocking=True)
        logitA, logitB = model(x_num, x_cat)
        allA_p.append(torch.sigmoid(logitA).cpu().numpy())
        allB_p.append(torch.sigmoid(logitB).cpu().numpy())
        allA_y.append(yA.numpy())
        allB_y.append(yB.numpy())
    A_p = np.concatenate(allA_p); A_y = np.concatenate(allA_y)
    B_p = np.concatenate(allB_p); B_y = np.concatenate(allB_y)
    return roc_auc_score(A_y, A_p), roc_auc_score(B_y, B_p)

def main():
    data = np.load("adult_stage1.npz", allow_pickle=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("device:", device)

    train_ds = AdultDataset(data["X_train_num"], data["X_train_cat"], data["yA_train"], data["yB_train"])
    val_ds   = AdultDataset(data["X_val_num"], data["X_val_cat"], data["yA_val"], data["yB_val"])

    train_loader = DataLoader(train_ds, batch_size=1024, shuffle=True, num_workers=2, pin_memory=True)
    val_loader   = DataLoader(val_ds, batch_size=4096, shuffle=False, num_workers=2, pin_memory=True)

    model = MTMLModel(num_dim=data["X_train_num"].shape[1], cat_sizes=data["cat_sizes"]).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)

    # Training targets (multi-task learning):
    # - Task A (yA): Primary target with weight 1.0 (income prediction - main focus)
    # - Task B (yB): Secondary target with weight 0.5 (auxiliary task - lower priority)
    wA, wB = 1.0, 0.5

    for epoch in range(1, 6):
        model.train()
        pbar = tqdm(train_loader, desc=f"epoch {epoch}")
        for x_num, x_cat, yA, yB in pbar:
            x_num = x_num.to(device, non_blocking=True)
            x_cat = x_cat.to(device, non_blocking=True)
            yA = yA.to(device, non_blocking=True)
            yB = yB.to(device, non_blocking=True)

            logitA, logitB = model(x_num, x_cat)
            # Binary cross-entropy loss for each task
            lossA = bce_logits(logitA, yA)  # Loss for Task A (income prediction)
            lossB = bce_logits(logitB, yB)  # Loss for Task B (secondary target)
            # Weighted combination: Task A is prioritized (weight 1.0 vs 0.5)
            loss = wA * lossA + wB * lossB

            opt.zero_grad()
            loss.backward()
            opt.step()
            pbar.set_postfix(loss=float(loss), lossA=float(lossA), lossB=float(lossB))

        aucA, aucB = eval_auc(model, val_loader, device)
        print(f"val AUC: taskA={aucA:.4f} taskB={aucB:.4f}")

    ckpt = {
    "model_state": model.state_dict(),
    "config": {
        "num_dim": int(data["X_train_num"].shape[1]),
        "cat_sizes": data["cat_sizes"].tolist(),
        "emb_dim": 16,
        "hidden": [256, 128, 64],
        "dropout": 0.1,
        }
    }
    torch.save(ckpt, "adult_mtml.pt")
    print("Saved adult_mtml.pt")


if __name__ == "__main__":
    main()
