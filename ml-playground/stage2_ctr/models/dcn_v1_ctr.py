import torch
import torch.nn as nn


class CrossNetV1(nn.Module):
    """
    CrossNet v1:
      x_{l+1} = x_0 * (w_l^T x_l) + b_l + x_l
    where:
      x_0, x_l are [B, D]
      w_l is [D], b_l is [D]
    """
    def __init__(self, input_dim: int, num_layers: int = 3):
        super().__init__()
        self.num_layers = num_layers
        self.w = nn.ParameterList([nn.Parameter(torch.randn(input_dim) * 0.01) for _ in range(num_layers)])
        self.b = nn.ParameterList([nn.Parameter(torch.zeros(input_dim)) for _ in range(num_layers)])

    def forward(self, x0: torch.Tensor) -> torch.Tensor:
        xl = x0
        for i in range(self.num_layers):
            # w^T x_l -> [B]
            xw = torch.sum(xl * self.w[i], dim=1, keepdim=True)  # [B, 1]
            # x0 * (w^T x_l) -> [B, D]
            cross = x0 * xw
            xl = cross + self.b[i] + xl
        return xl


class DCNV1CTR(nn.Module):
    """
    DCN v1 for CTR:
      - embedding lookup for sparse
      - concat with dense -> x0
      - CrossNet(x0) -> x_cross
      - MLP(x0) -> x_deep
      - concat [x_cross, x_deep] -> final linear -> logit
    """
    def __init__(
        self,
        num_embeddings: int,
        embed_dim: int = 16,
        num_dense: int = 13,
        num_sparse_fields: int = 26,
        cross_layers: int = 3,
        deep_hidden_dims=(256, 128, 64),
        dropout: float = 0.0,
    ):
        super().__init__()
        self.num_sparse_fields = num_sparse_fields
        self.embed_dim = embed_dim

        self.embed = nn.Embedding(num_embeddings=num_embeddings, embedding_dim=embed_dim)
        nn.init.normal_(self.embed.weight, mean=0.0, std=0.01)

        self.input_dim = num_dense + num_sparse_fields * embed_dim

        self.crossnet = CrossNetV1(input_dim=self.input_dim, num_layers=cross_layers)

        # Deep tower (MLP) on x0
        layers = []
        prev = self.input_dim
        for h in deep_hidden_dims:
            layers.append(nn.Linear(prev, h))
            layers.append(nn.ReLU())
            if dropout and dropout > 0:
                layers.append(nn.Dropout(dropout))
            prev = h
        self.deep = nn.Sequential(*layers)

        # Final
        final_in = self.input_dim + (deep_hidden_dims[-1] if len(deep_hidden_dims) > 0 else self.input_dim)
        self.fc_out = nn.Linear(final_in, 1)

    def forward(self, dense, sparse):
        e = self.embed(sparse)                 # [B, 26, D]
        e = e.reshape(e.size(0), -1)           # [B, 26*D]
        x0 = torch.cat([dense, e], dim=1)      # [B, input_dim]

        x_cross = self.crossnet(x0)            # [B, input_dim]
        x_deep = self.deep(x0) if len(self.deep) > 0 else x0

        x = torch.cat([x_cross, x_deep], dim=1)
        logit = self.fc_out(x).squeeze(1)      # [B]
        return logit
