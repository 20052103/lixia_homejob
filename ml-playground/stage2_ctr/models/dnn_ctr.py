import torch
import torch.nn as nn


class DNNCTR(nn.Module):
    """
    最简单 CTR DNN:
      - sparse: 一个大 embedding table（index 是 global_id）
      - dense: 13维连续特征
      - 拼接: [dense, flatten(embeddings)]
      - MLP -> logit
    """

    def __init__(
        self,
        num_embeddings: int,
        embed_dim: int = 16,
        num_dense: int = 13,
        hidden_dims=(256, 128, 64),
        dropout: float = 0.0,
    ):
        super().__init__()
        self.embed = nn.Embedding(num_embeddings=num_embeddings, embedding_dim=embed_dim)

        in_dim = num_dense + 26 * embed_dim

        layers = []
        prev = in_dim
        for h in hidden_dims:
            layers.append(nn.Linear(prev, h))
            layers.append(nn.ReLU())
            if dropout and dropout > 0:
                layers.append(nn.Dropout(dropout))
            prev = h

        layers.append(nn.Linear(prev, 1))  # logit
        self.mlp = nn.Sequential(*layers)

        # 初始化（可选）：embedding 小一点更稳定
        nn.init.normal_(self.embed.weight, mean=0.0, std=0.01)

    def forward(self, dense, sparse):
        """
        dense: [B, 13] float
        sparse: [B, 26] long (global ids)
        """
        e = self.embed(sparse)              # [B, 26, D]
        e = e.reshape(e.size(0), -1)        # [B, 26*D]
        x = torch.cat([dense, e], dim=1)    # [B, 13 + 26*D]
        logit = self.mlp(x).squeeze(1)      # [B]
        return logit
