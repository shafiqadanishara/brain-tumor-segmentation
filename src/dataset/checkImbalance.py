import os
import numpy as np
from tqdm import tqdm
from src.dataset.dataset3D import BraTSDataset3D

dataset = BraTSDataset3D("data/split/train")

wt_total = 0
tc_total = 0
et_total = 0
voxel_total = 0

for i in tqdm(range(len(dataset))):
    _, mask = dataset[i]   # mask shape: (3, D, H, W)

    wt = mask[0]
    tc = mask[1]
    et = mask[2]

    wt_total += wt.sum()
    tc_total += tc.sum()
    et_total += et.sum()

    voxel_total += np.prod(wt.shape)

# convert to %
wt_ratio = wt_total / voxel_total
tc_ratio = tc_total / voxel_total
et_ratio = et_total / voxel_total

print("\n=== Class Distribution ===")
print(f"WT: {wt_ratio:.6f}")
print(f"TC: {tc_ratio:.6f}")
print(f"ET: {et_ratio:.6f}")
print(f"Background: {1 - (wt_ratio + tc_ratio + et_ratio):.6f}")