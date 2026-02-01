import numpy as np
import pandas as pd
import pickle
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# =========================
# 1. Load raw Adult data
# =========================
cols = [
    "age","workclass","fnlwgt","education","education-num",
    "marital-status","occupation","relationship","race","sex",
    "capital-gain","capital-loss","hours-per-week","native-country","income"
]

df = pd.read_csv(
    "data/adult.data",
    names=cols,
    sep=",",
    skipinitialspace=True,
    na_values="?"
)

# =========================
# 2. Define tasks (labels)
# =========================
# Task A: income >50K
yA = (
    df["income"]
    .astype(str)
    .str.contains(">50K")
    .astype(np.float32)
    .values
)

# Task B: married?
yB = (
    df["marital-status"]
    .fillna("__MISSING__")
    .astype(str)
    .str.startswith("Married")
    .astype(np.float32)
    .values
)

# =========================
# 3. Feature columns
# =========================
num_cols = [
    "age", "fnlwgt", "education-num",
    "capital-gain", "capital-loss", "hours-per-week"
]

cat_cols = [
    "workclass", "education", "marital-status",
    "occupation", "relationship", "race",
    "sex", "native-country"
]

# =========================
# 4. Missing value handling
# =========================
df[num_cols] = df[num_cols].fillna(0.0)
df[cat_cols] = df[cat_cols].fillna("__MISSING__")

X = df[num_cols + cat_cols].copy()

# =========================
# 5. Train / Val / Test split
# =========================
X_train, X_tmp, yA_train, yA_tmp, yB_train, yB_tmp = train_test_split(
    X, yA, yB,
    test_size=0.30,
    random_state=42,
    stratify=yA
)

X_val, X_test, yA_val, yA_test, yB_val, yB_test = train_test_split(
    X_tmp, yA_tmp, yB_tmp,
    test_size=0.50,
    random_state=42,
    stratify=yA_tmp
)

# =========================
# 6. Numeric preprocessing
# =========================
scaler = StandardScaler()

X_train_num = scaler.fit_transform(
    X_train[num_cols].values
).astype(np.float32)

X_val_num = scaler.transform(
    X_val[num_cols].values
).astype(np.float32)

X_test_num = scaler.transform(
    X_test[num_cols].values
).astype(np.float32)

# =========================
# 7. Categorical encoding
# =========================
def build_mapping(series: pd.Series):
    """Build value -> index mapping"""
    cats = pd.Index(series.unique())
    return {k: i for i, k in enumerate(cats)}

X_train_cat, X_val_cat, X_test_cat = [], [], []
cat_sizes = []

# ---- inference-time preprocessing info ----
preproc = {
    "num_cols": num_cols,
    "cat_cols": cat_cols,
    "scaler_mean": scaler.mean_.astype(np.float32),
    "scaler_scale": scaler.scale_.astype(np.float32),
    "cat_mappings": {},     # col -> {value: index}
    "cat_oov_index": {},    # col -> oov index
}

for c in cat_cols:
    mapping = build_mapping(X_train[c])
    oov = len(mapping)

    preproc["cat_mappings"][c] = mapping
    preproc["cat_oov_index"][c] = oov
    cat_sizes.append(oov + 1)

    def encode(series):
        arr = (
            series
            .map(lambda x: mapping.get(x, -1))
            .astype(np.int64)
            .values
        )
        arr = np.where(arr < 0, oov, arr)
        return arr

    X_train_cat.append(encode(X_train[c]))
    X_val_cat.append(encode(X_val[c]))
    X_test_cat.append(encode(X_test[c]))

X_train_cat = np.stack(X_train_cat, axis=1)
X_val_cat   = np.stack(X_val_cat, axis=1)
X_test_cat  = np.stack(X_test_cat, axis=1)

# =========================
# 8. Save training arrays
# =========================
np.savez(
    "adult_stage1.npz",
    X_train_num=X_train_num,
    X_val_num=X_val_num,
    X_test_num=X_test_num,
    X_train_cat=X_train_cat,
    X_val_cat=X_val_cat,
    X_test_cat=X_test_cat,
    yA_train=yA_train,
    yA_val=yA_val,
    yA_test=yA_test,
    yB_train=yB_train,
    yB_val=yB_val,
    yB_test=yB_test,
    cat_sizes=np.array(cat_sizes, dtype=np.int64),
)

# =========================
# 9. Save inference preproc
# =========================
with open("adult_preproc.pkl", "wb") as f:
    pickle.dump(preproc, f)

# =========================
# 10. Print summary
# =========================
print("Saved files:")
print("  - adult_stage1.npz")
print("  - adult_preproc.pkl")
print()
print("Train / Val / Test sizes:",
      len(yA_train), len(yA_val), len(yA_test))
print("Positive rate (Task A, train):", float(yA_train.mean()))
print("cat_sizes:", cat_sizes)
