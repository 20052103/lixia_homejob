import torch
from models.dcn_v2_ctr import DCNV2CTR

B=4
num_embeddings=52000000
dense=torch.randn(B,13)
sparse=torch.randint(0, num_embeddings, (B,26), dtype=torch.long)

m=DCNV2CTR(num_embeddings=num_embeddings, embed_dim=16, cross_layers=3, cross_rank=32, cross_experts=4)
logit=m(dense, sparse)
print("logit shape:", logit.shape)
