import sys
import torch
import numpy as np
import pandas as pd
import sklearn
import tqdm

print("python:", sys.version.split()[0])
print("torch:", torch.__version__)
print("cuda available:", torch.cuda.is_available())
if torch.cuda.is_available():
    print("gpu:", torch.cuda.get_device_name(0))
    print("cap:", torch.cuda.get_device_capability(0))

print("numpy:", np.__version__)
print("pandas:", pd.__version__)
print("sklearn:", sklearn.__version__)
print("tqdm:", tqdm.__version__)
print("OK ✅")
