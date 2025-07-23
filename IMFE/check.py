import numpy as np
import os

all_feats = []

load_count = 0
for file in os.listdir("/data1/ghufran/helping_python_scripts/IMFE_train"):
    if file.endswith(".npy"):
        feat = np.load(os.path.join("/data1/ghufran/helping_python_scripts/IMFE_train", file))  # shape: [N, 2048]
        all_feats.append(feat)
        print(f"Loaded {file} with shape {feat.shape}")
        load_count += 1
        if load_count >= 100:
            print("Loaded 50 files, stopping.")
            break

all_feats = np.concatenate(all_feats, axis=0)  # [total_clips, 2048]

print("Mean:", np.mean(all_feats))
print("Std:", np.std(all_feats))
print("Min:", np.min(all_feats))
print("Max:", np.max(all_feats))
