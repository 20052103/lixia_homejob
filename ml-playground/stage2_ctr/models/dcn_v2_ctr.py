import torch
import torch.nn as nn


class DCNv2CrossLayer(nn.Module):
    """
    DCNv2-style cross layer (low-rank, with optional mixture-of-experts):

    For each layer:
      x_{l+1} = x_l + x0 * cross(x_l)

    cross(x_l) is computed by:
      - (Option A) Low-rank:  U( V^T x_l ) + b, where V: [D, r], U: [r, D]
      - (Option B) MoE: sum_e g_e(x_l) * (U_e(V_e^T x_l) + b_e)

    Shapes:
      x0, x_l: [B, D]
      output:  [B, D]
    """
    def __init__(self, input_dim: int, rank: int = 32, num_experts: int = 4):
        super().__init__()
        self.input_dim = input_dim
        self.rank = rank
        self.num_experts = num_experts

        # Experts parameters
        self.V = nn.Parameter(torch.randn(num_experts, input_dim, rank) * 0.01)   # [E, D, r]
        self.U = nn.Parameter(torch.randn(num_experts, rank, input_dim) * 0.01)  # [E, r, D]
        self.b = nn.Parameter(torch.zeros(num_experts, input_dim))               # [E, D]

        # Gate: map x_l -> weights over experts
        self.gate = nn.Linear(input_dim, num_experts)

    def forward(self, x0: torch.Tensor, xl: torch.Tensor) -> torch.Tensor:
        B, D = xl.shape
        assert D == self.input_dim

        if self.num_experts == 1:
            # low-rank (no mixture)
            v = torch.matmul(xl, self.V[0])          # [B, r]
            u = torch.matmul(v, self.U[0])           # [B, D]
            cross = u + self.b[0]                    # [B, D]
            return xl + x0 * cross

        # MoE weights
        g = torch.softmax(self.gate(xl), dim=1)      # [B, E]

        # Compute each expert output: [B, D]
        # We'll accumulate efficiently in a loop over experts (E is small).
        out = xl
        for e in range(self.num_experts):
            v = torch.matmul(xl, self.V[e])          # [B, r]
            u = torch.matmul(v, self.U[e])           # [B, D]
            cross_e = u + self.b[e]                  # [B, D]
            out = out + (g[:, e:e+1] * (x0 * cross_e))
        return out


class DCNv2CrossNet(nn.Module):
    def __init__(self, input_dim: int, num_layers: int = 3, rank: int = 32, num_experts: int = 4):
        super().__init__()
        self.layers = nn.ModuleList([
            DCNv2CrossLayer(input_dim=input_dim, rank=rank, num_experts=num_experts)
            for _ in range(num_layers)
        ])

    def forward(self, x0: torch.Tensor) -> torch.Tensor:
        xl = x0
        for layer in self.layers:
            xl = layer(x0, xl)
        return xl


class DCNV2CTR(nn.Module):
    """
    DCNv2 CTR model:
      x0 = concat(dense, embed_flat)
      x_cross = CrossNetV2(x0)
      x_deep = MLP(x0)
      concat -> final linear -> logit
    """
    def __init__(
        self,
        num_embeddings: int,
        embed_dim: int = 16,
        num_dense: int = 13,
        num_sparse_fields: int = 26,
        cross_layers: int = 3,
        cross_rank: int = 32,
        cross_experts: int = 4,
        deep_hidden_dims=(256, 128, 64),
        dropout: float = 0.0,
    ):
        super().__init__()
        self.num_sparse_fields = num_sparse_fields
        self.embed = nn.Embedding(num_embeddings=num_embeddings, embedding_dim=embed_dim)
        nn.init.normal_(self.embed.weight, mean=0.0, std=0.01)

        self.input_dim = num_dense + num_sparse_fields * embed_dim

        self.crossnet = DCNv2CrossNet(
            input_dim=self.input_dim,
            num_layers=cross_layers,
            rank=cross_rank,
            num_experts=cross_experts,
        )

        # Deep tower
        layers = []
        prev = self.input_dim
        for h in deep_hidden_dims:
            layers.append(nn.Linear(prev, h))
            layers.append(nn.ReLU())
            if dropout and dropout > 0:
                layers.append(nn.Dropout(dropout))
            prev = h
        self.deep = nn.Sequential(*layers)

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
