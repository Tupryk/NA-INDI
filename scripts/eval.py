import os
import numpy as np

data_path = "outputs/2025-11-04"
models = os.listdir(data_path)
mins = np.ones(6) * 1e6
mins_names = ["" for _ in range(6)]

for model in models:
    dir = os.path.join(data_path, model, "eval.txt")
    with open(dir, "r") as f:
        data = [float(x) for x in f.read().split()]
    for j in range(6):
        if data[j] < mins[j]:
            mins[j] = data[j]
            mins_names[j] = model

print(mins)
print(mins_names)
    