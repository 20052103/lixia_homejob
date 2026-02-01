import pickle
import numpy as np
import torch
import torch.nn as nn

# ---- model (same as training) ----
class MTMLModel(nn.Module):
    def __init__(self, num_dim, cat_sizes, emb_dim=16, hidden=(256,128,64), dropout=0.1):
        super().__init__()
        self.embs = nn.ModuleList([nn.Embedding(int(sz), emb_dim) for sz in cat_sizes])
        in_dim = num_dim + emb_dim * len(cat_sizes)

        layers = []
        d = in_dim
        for h in hidden:
            layers += [nn.Linear(d, h), nn.ReLU(), nn.Dropout(dropout)]
            d = h
        self.shared = nn.Sequential(*layers)
        self.headA = nn.Linear(d, 1)  # income >50K
        self.headB = nn.Linear(d, 1)  # married?

    def forward(self, x_num, x_cat):
        emb_list = [emb(x_cat[:, i]) for i, emb in enumerate(self.embs)]
        x = torch.cat([x_num] + emb_list, dim=1)
        h = self.shared(x)
        return self.headA(h).squeeze(1), self.headB(h).squeeze(1)

def standardize(x, mean, scale):
    return (x - mean) / scale

def encode_examples(examples, preproc):
    num_cols = preproc["num_cols"]
    cat_cols = preproc["cat_cols"]

    mean = preproc["scaler_mean"]
    scale = preproc["scaler_scale"]

    # numeric
    X_num = []
    for ex in examples:
        X_num.append([float(ex.get(c, 0.0)) for c in num_cols])
    X_num = np.array(X_num, dtype=np.float32)
    X_num = standardize(X_num, mean, scale).astype(np.float32)

    # categorical
    X_cat = []
    for ex in examples:
        row = []
        for c in cat_cols:
            val = str(ex.get(c, "__MISSING__"))
            mapping = preproc["cat_mappings"][c]
            oov = preproc["cat_oov_index"][c]
            row.append(mapping.get(val, oov))
        X_cat.append(row)
    X_cat = np.array(X_cat, dtype=np.int64)

    return X_num, X_cat

def main():
    # load preproc
    with open("adult_preproc.pkl", "rb") as f:
        preproc = pickle.load(f)

    # load model
    ckpt = torch.load("adult_mtml.pt", map_location="cpu")
    cfg = ckpt["config"]

    model = MTMLModel(
        num_dim=cfg["num_dim"],
        cat_sizes=cfg["cat_sizes"],
        emb_dim=cfg["emb_dim"],
        hidden=tuple(cfg["hidden"]),
        dropout=cfg["dropout"],
    )
    model.load_state_dict(ckpt["model_state"])
    model.eval()

    # ---- handcrafted examples ----
    examples = [
        {
            "age": 39, "workclass": "State-gov", "fnlwgt": 77516, "education": "Bachelors",
            "education-num": 13, "marital-status": "Never-married", "occupation": "Adm-clerical",
            "relationship": "Not-in-family", "race": "White", "sex": "Male",
            "capital-gain": 0, "capital-loss": 0, "hours-per-week": 40, "native-country": "United-States",
        },
        {
            "age": 52, "workclass": "Self-emp-inc", "fnlwgt": 120000, "education": "Masters",
            "education-num": 14, "marital-status": "Married-civ-spouse", "occupation": "Exec-managerial",
            "relationship": "Husband", "race": "White", "sex": "Male",
            "capital-gain": 15000, "capital-loss": 0, "hours-per-week": 60, "native-country": "United-States",
        },
        {
            "age": 28, "workclass": "Private", "fnlwgt": 210000, "education": "HS-grad",
            "education-num": 9, "marital-status": "Divorced", "occupation": "Sales",
            "relationship": "Unmarried", "race": "Black", "sex": "Female",
            "capital-gain": 0, "capital-loss": 0, "hours-per-week": 35, "native-country": "United-States",
        },
    ]

    X_num, X_cat = encode_examples(examples, preproc)

    with torch.no_grad():
        x_num_t = torch.from_numpy(X_num)
        x_cat_t = torch.from_numpy(X_cat)
        logitA, logitB = model(x_num_t, x_cat_t)
        p_income = torch.sigmoid(logitA).numpy()
        p_married = torch.sigmoid(logitB).numpy()

    for i, ex in enumerate(examples):
        print(f"\nExample #{i+1}")
        print("  raw:", {k: ex[k] for k in ["age","education","occupation","hours-per-week","marital-status"]})
        print(f"  P(income>50K) = {p_income[i]:.4f}")
        print(f"  P(married)    = {p_married[i]:.4f}")

if __name__ == "__main__":
    main()
