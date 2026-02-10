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
        # Embedding 层：将分类特征（整数索引）转换为稠密向量
        # cat_sizes = [9, 17, 8, 15, 7, 6, 3, 43]  (每列的唯一值个数 + 1)
        # 创建 8 个 Embedding 层，每个负责一个分类特征
        # 例如：workclass (9个值) → 9维 embedding → 16维向量
        self.embs = nn.ModuleList([nn.Embedding(int(sz), emb_dim) for sz in cat_sizes])
        
        # 计算合并后的特征维度
        # = 数值特征维度 + (分类特征数 × embedding维度)
        # = 6 + (8 × 16) = 134
        in_dim = num_dim + emb_dim * len(cat_sizes)

        # Shared layers: 共享的多层神经网络（特征提取器）
        # 目的：为两个任务学习通用的特征表示
        layers = []
        d = in_dim  # d = 134（合并后的特征维度）
        for h in hidden:  # hidden = (256, 128, 64)
            # 每次迭代添加：Linear → ReLU → Dropout 三层
            # 第1次：Linear(134, 256) → ReLU → Dropout
            # 第2次：Linear(256, 128) → ReLU → Dropout
            # 第3次：Linear(128, 64) → ReLU → Dropout
            layers += [nn.Linear(d, h), nn.ReLU(), nn.Dropout(dropout)]
            d = h  # 更新维度用于下一层输入
        
        # nn.Sequential(*layers) 将所有层串联成一个完整的网络
        # *layers 解包列表，相当于：
        # nn.Sequential(
        #     Linear(134, 256), ReLU, Dropout,
        #     Linear(256, 128), ReLU, Dropout,
        #     Linear(128, 64), ReLU, Dropout
        # )
        self.shared = nn.Sequential(*layers)

        # Task-specific output heads for each target
        self.headA = nn.Linear(d, 1)  # Head for Task A (income prediction)
        self.headB = nn.Linear(d, 1)  # Head for Task B (secondary target)

    def forward(self, x_num, x_cat):
        # emb_list：对每个分类特征进行 embedding
        # x_cat.shape = (batch_size, 8)  ← 8列分类特征
        
        # enumerate(self.embs) 遍历 self.embs 中的每个 embedding 层
        # self.embs 在 __init__ 中定义：
        #   self.embs = nn.ModuleList([
        #       nn.Embedding(9, 16),    # i=0, emb 是这个 embedding 层
        #       nn.Embedding(17, 16),   # i=1, emb 是这个 embedding 层
        #       ...
        #       nn.Embedding(43, 16)    # i=7, emb 是这个 embedding 层
        #   ])
        
        # 对每个 embedding 层 emb 和对应的列索引 i：
        #   x_cat[:, i] → (batch_size,) 该列的所有值
        #   emb(x_cat[:, i]) → (batch_size, 16) embedding 后的向量
        
        emb_list = [emb(x_cat[:, i]) for i, emb in enumerate(self.embs)]
        # emb_list = [
        #     (batch_size, 16),  # workclass 的 embedding
        #     (batch_size, 16),  # education 的 embedding
        #     ...
        #     (batch_size, 16)   # native-country 的 embedding
        # ]
        
        # 合并数值特征和 embedding 特征
        # x_num: (batch_size, 6)
        # emb_list: 8 个 (batch_size, 16)
        # 结果：(batch_size, 6 + 8*16) = (batch_size, 134)
        x = torch.cat([x_num] + emb_list, dim=1)
        
        # 通过共享层提取特征
        h = self.shared(x)  # (batch_size, 64)
        
        # 两个任务头分别预测
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

            # ===== 梯度下降的三个关键步骤 =====
            
            # Step 1: 清空之前的梯度
            # 为什么需要？PyTorch 默认累加梯度，如果不清空，新梯度会加到旧梯度上
            # opt.zero_grad() 将所有参数的梯度设为 0
            opt.zero_grad()
            
            # Step 2: 反向传播计算梯度
            # loss.backward() 从当前 loss 开始，计算所有参数相对于 loss 的梯度
            # 这个过程涉及链式法则，从输出层逐层反向计算
            # 梯度被存储在每个参数的 .grad 属性中
            loss.backward()
            
            # Step 3: 用梯度更新参数
            # opt.step() 根据计算得到的梯度，用优化器算法更新模型参数
            # 使用 AdamW 优化器，学习率 lr=1e-3
            # 更新公式（简化）：param = param - lr * param.grad
            opt.step()
            
            # 进度条显示当前 batch 的 loss
            # loss：总的加权 loss
            # lossA：Task A 的 loss
            # lossB：Task B 的 loss
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
