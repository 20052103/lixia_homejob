import torch
import torch.nn as nn

class TabTokenizer(nn.Module):
    """
    Turn (dense: [B, Dd], sparse: [B, Dc]) into tokens: [B, F, d_token]
    - sparse: integer ids per categorical field
    - dense: float values
    Supports:
      - dense_token_mode="per_feature": Dd dense features -> Dd tokens
      - dense_token_mode="single":      all dense -> 1 token
    """
    def __init__(
        self,
        num_dense: int,
        num_cat: int,
        cat_cardinalities: list[int],   # length = num_cat
        d_token: int,
        dense_token_mode: str = "single",  # "single" or "per_feature"
        use_cls: bool = False,
    ):
        super().__init__()
        assert len(cat_cardinalities) == num_cat
        assert dense_token_mode in ("single", "per_feature")

        self.num_dense = num_dense
        self.num_cat = num_cat
        self.d_token = d_token
        self.dense_token_mode = dense_token_mode
        self.use_cls = use_cls

        # One embedding table per field (simple + clear)
        self.cat_embs = nn.ModuleList([
            nn.Embedding(card, d_token) for card in cat_cardinalities
        ])

        # Dense -> token(s)
        if dense_token_mode == "single":
            self.dense_proj = nn.Linear(num_dense, d_token)
        else:
            # Each dense feature has a learnable vector (FiLM-like): token_i = x_i * w_i + b_i
            self.dense_w = nn.Parameter(torch.randn(num_dense, d_token) * 0.02)
            self.dense_b = nn.Parameter(torch.zeros(num_dense, d_token))

        if use_cls:
            self.cls = nn.Parameter(torch.zeros(1, 1, d_token))
        else:
            self.cls = None

    def forward(self, dense: torch.Tensor, sparse: torch.Tensor) -> torch.Tensor:
        """
        dense:  [B, Dd] float
        sparse: [B, Dc] long/int
        return: [B, F, d_token]
        """
        B = dense.size(0)

        # cat tokens: list of [B, d_token] -> stack -> [B, Dc, d_token]
        # sparse[:, i] must be within [0, card_i)
        cat_tokens = []
        for i in range(self.num_cat):
            cat_tokens.append(self.cat_embs[i](sparse[:, i].long()))
        cat_tokens = torch.stack(cat_tokens, dim=1)

        # dense tokens
        if self.dense_token_mode == "single":
            dense_tokens = self.dense_proj(dense).unsqueeze(1)  # [B, 1, d_token]
        else:
            # [B, Dd, 1] * [Dd, d_token] -> [B, Dd, d_token]
            dense_tokens = dense.unsqueeze(-1) * self.dense_w.unsqueeze(0) + self.dense_b.unsqueeze(0)

        tokens = torch.cat([cat_tokens, dense_tokens], dim=1)  # [B, Dc + (1 or Dd), d_token]

        if self.use_cls:
            cls = self.cls.expand(B, -1, -1)  # [B, 1, d_token]
            tokens = torch.cat([cls, tokens], dim=1)

        return tokens
