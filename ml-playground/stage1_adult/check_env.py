import sys
print("python:", sys.version)

import torch
print("torch:", torch.__version__)
print("cuda:", torch.version.cuda)
print("cuda available:", torch.cuda.is_available())
if torch.cuda.is_available():
    print("gpu:", torch.cuda.get_device_name(0))
    print("cap:", torch.cuda.get_device_capability(0))

import numpy, pandas, sklearn, tqdm
print("numpy:", numpy.__version__)
print("pandas:", pandas.__version__)
print("sklearn:", sklearn.__version__)

import ucimlrepo
print("ucimlrepo: import ok")


print("OK ✅")
