import torch
import torch.nn as nn

from .tab_tokenizer import TabTokenizer

class TransformerBlock(nn.Module):
    def __init__(self, d_token: int, n_heads: int, ffn_mult: int = 2, dropout: float = 0.1):
        super().__init__()
        self.ln1 = nn.LayerNorm(d_token)
        self.attn = nn.MultiheadAttention(embed_dim=d_token, num_heads=n_heads, dropout=dropout, batch_first=True)
        self.drop1 = nn.Dropout(dropout)

        self.ln2 = nn.LayerNorm(d_token)
        self.ffn = nn.Sequential(
            nn.Linear(d_token, ffn_mult * d_token),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(ffn_mult * d_token, d_token),
        )
        self.drop2 = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # PreNorm attention
        h = self.ln1(x)
        attn_out, _ = self.attn(h, h, h, need_weights=False)
        x = x + self.drop1(attn_out)

        # PreNorm FFN
        h = self.ln2(x)
        x = x + self.drop2(self.ffn(h))
        return x

class AutoIntCTR(nn.Module):
    def __init__(
        self,
        num_dense: int,
        num_cat: int,
        cat_cardinalities: list[int],
        d_token: int = 64,
        n_heads: int = 4,
        n_layers: int = 2,
        ffn_mult: int = 2,
        dropout: float = 0.1,
        dense_token_mode: str = "single",   # "single" recommended first
        pooling: str = "mean",              # "mean" or "cls"
    ):
        super().__init__()
        assert pooling in ("mean", "cls")
        use_cls = (pooling == "cls")

        self.tokenizer = TabTokenizer(
            num_dense=num_dense,
            num_cat=num_cat,
            cat_cardinalities=cat_cardinalities,
            d_token=d_token,
            dense_token_mode=dense_token_mode,
            use_cls=use_cls,
        )

        self.blocks = nn.ModuleList([
            TransformerBlock(d_token, n_heads, ffn_mult=ffn_mult, dropout=dropout)
            for _ in range(n_layers)
        ])
        self.ln_out = nn.LayerNorm(d_token)

        self.head = nn.Sequential(
            nn.Linear(d_token, d_token),
            nn.ReLU(),
            nn.Linear(d_token, 1),
        )

        self.pooling = pooling

    def forward(self, dense: torch.Tensor, sparse: torch.Tensor) -> torch.Tensor:
        x = self.tokenizer(dense, sparse)  # [B, F, d_token]
        for blk in self.blocks:
            x = blk(x)
        x = self.ln_out(x)

        if self.pooling == "cls":
            pooled = x[:, 0, :]          # [B, d_token]
        else:
            pooled = x.mean(dim=1)       # [B, d_token]

        logit = self.head(pooled).squeeze(-1)  # [B]
        return logit
