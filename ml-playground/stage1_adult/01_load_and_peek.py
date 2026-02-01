import pandas as pd

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

print("df shape:", df.shape)
print(df.head(3))
print("\nincome counts:\n", df["income"].value_counts())
print("\nmissing by col:\n", df.isna().sum().sort_values(ascending=False).head(10))
