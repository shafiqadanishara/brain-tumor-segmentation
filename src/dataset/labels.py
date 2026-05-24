import nibabel as nib
import numpy as np
import os

print("\n=== UPenn GBM ===")
segm_dir = 'data/UPENN_GBM/images_segm'
seg_files = sorted(os.listdir(segm_dir))[:5]
for f in seg_files:
    seg = nib.load(os.path.join(segm_dir, f)).get_fdata()
    counts = {int(v): int((seg==v).sum()) for v in np.unique(seg) if v > 0}
    print(f'  {f}: {counts}')