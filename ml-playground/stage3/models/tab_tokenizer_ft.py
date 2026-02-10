import torch
import torch.nn as nn

class TabTokenizerFT(nn.Module):
    """
    FT-Transformer style tokenizer:
      - Categorical fields: shared embedding table [hash_bins, d_token]
      - Dense features: projected to a single token via Linear(13 -> d_token) by default
      - Output tokens: [B, F, d_token], where F = num_cat + num_dense_tokens (+1 cls if use_cls)
    """
    def __init__(
        self,
        num_dense: int,
        num_cat: int,
        hash_bins: int,
        d_token: int,
        dense_token_mode: str = "single",  # "single" or "per_feature"
        use_cls: bool = True,
    ):
        super().__init__()
        assert dense_token_mode in ("single", "per_feature")
        self.num_dense = num_dense
        self.num_cat = num_cat
        self.hash_bins = int(hash_bins)
        self.d_token = int(d_token)
        self.dense_token_mode = dense_token_mode
        self.use_cls = use_cls

        # Shared cat embedding: [hash_bins, d_token]
        self.cat_emb = nn.Embedding(self.hash_bins, self.d_token)

        # Dense -> token(s)
        if self.dense_token_mode == "single":
            self.dense_proj = nn.Linear(self.num_dense, self.d_token)
        else:
            # per-feature: token_i = x_i * w_i + b_i
            self.dense_w = nn.Parameter(torch.randn(self.num_dense, self.d_token) * 0.02)
            self.dense_b = nn.Parameter(torch.zeros(self.num_dense, self.d_token))

        if self.use_cls:
            self.cls = nn.Parameter(torch.zeros(1, 1, self.d_token))
        else:
            self.cls = None

    def forward(self, dense: torch.Tensor, sparse: torch.Tensor) -> torch.Tensor:
        """
        dense:  [B, 13] float32
        sparse: [B, 26] long, values must be in [0, hash_bins)
        return: [B, F, d_token]
        """
        B = dense.size(0)

        # [B, 26, d]
        cat_tokens = self.cat_emb(sparse.long())

        # dense tokens
        if self.dense_token_mode == "single":
            dense_tokens = self.dense_proj(dense).unsqueeze(1)  # [B, 1, d]
        else:
            dense_tokens = dense.unsqueeze(-1) * self.dense_w.unsqueeze(0) + self.dense_b.unsqueeze(0)  # [B, 13, d]

        tokens = torch.cat([cat_tokens, dense_tokens], dim=1)  # [B, 26+(1 or 13), d]

        if self.use_cls:
            cls = self.cls.expand(B, -1, -1)  # [B, 1, d]
            tokens = torch.cat([cls, tokens], dim=1)

        return tokens
