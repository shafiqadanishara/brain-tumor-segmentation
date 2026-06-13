import os
import hashlib
import nibabel as nib
import numpy as np
from tqdm import tqdm

# =====================================================
# CONFIG
# =====================================================

BRATS_ROOT = "data/raw/ASNR-MICCAI-BraTS2023-GLI-Challenge-TrainingData"
UPENN_ROOT = "data_upenn/UPENN_GBM/images_structural"

OUTPUT_OVERLAP = "overlap_cases.txt"
OUTPUT_CLEAN   = "clean_upenn_cases.txt"

# =====================================================
# HASH FUNCTION
# =====================================================

def volume_hash(path):
    """
    Generate MD5 hash from voxel values.
    Identical MRI volumes -> identical hash.
    """

    vol = nib.load(path).get_fdata()

    # reduce memory
    vol = vol.astype(np.float32)

    return hashlib.md5(
        vol.tobytes()
    ).hexdigest()

# =====================================================
# LOAD BRATS HASHES
# =====================================================

print("\nLoading BraTS volumes...")

brats_hashes = {}

cases = sorted(os.listdir(BRATS_ROOT))

for case in tqdm(cases, desc="BraTS"):

    case_dir = os.path.join(BRATS_ROOT, case)

    if not os.path.isdir(case_dir):
        continue

    files = [
        f for f in os.listdir(case_dir)
        if "t1n" in f.lower()
    ]

    if len(files) == 0:
        continue

    path = os.path.join(case_dir, files[0])

    try:
        brats_hashes[case] = volume_hash(path)

    except Exception as e:
        print(f"Failed: {case}")
        print(e)

print(f"\nBraTS loaded: {len(brats_hashes)} cases")

# =====================================================
# LOAD UPENN HASHES
# =====================================================

print("\nLoading UPENN volumes...")

upenn_hashes = {}

cases = sorted(os.listdir(UPENN_ROOT))

for case in tqdm(cases, desc="UPENN"):

    case_dir = os.path.join(UPENN_ROOT, case)

    if not os.path.isdir(case_dir):
        continue

    path = os.path.join(
        case_dir,
        f"{case}_T1.nii.gz"
    )

    if not os.path.exists(path):
        continue

    try:
        upenn_hashes[case] = volume_hash(path)

    except Exception as e:
        print(f"Failed: {case}")
        print(e)

print(f"\nUPENN loaded: {len(upenn_hashes)} cases")

# =====================================================
# BUILD LOOKUP
# =====================================================

print("\nBuilding lookup table...")

hash_to_brats = {}

for case, h in brats_hashes.items():
    hash_to_brats[h] = case

# =====================================================
# MATCHING
# =====================================================

print("\nSearching overlaps...")

overlaps = []

for up_case, up_hash in tqdm(
    upenn_hashes.items(),
    total=len(upenn_hashes),
    desc="Matching"
):

    if up_hash in hash_to_brats:

        overlaps.append(
            (
                hash_to_brats[up_hash],
                up_case
            )
        )

# =====================================================
# RESULTS
# =====================================================

print("\n================================")
print("RESULT")
print("================================")

print(f"BraTS cases : {len(brats_hashes)}")
print(f"UPENN cases : {len(upenn_hashes)}")
print(f"Overlaps    : {len(overlaps)}")

# =====================================================
# SAVE OVERLAPS
# =====================================================

with open(OUTPUT_OVERLAP, "w") as f:

    for brats_case, upenn_case in overlaps:

        line = (
            f"BraTS: {brats_case}"
            f" <--> "
            f"UPENN: {upenn_case}\n"
        )

        f.write(line)

print(f"\nSaved overlap list -> {OUTPUT_OVERLAP}")

# =====================================================
# CLEAN CASES
# =====================================================

contaminated = {
    up_case
    for _, up_case in overlaps
}

clean_cases = sorted(
    set(upenn_hashes.keys()) - contaminated
)

with open(OUTPUT_CLEAN, "w") as f:

    for case in clean_cases:
        f.write(case + "\n")

print(f"Saved clean list -> {OUTPUT_CLEAN}")

# =====================================================
# SUMMARY
# =====================================================

print("\n================================")
print("SUMMARY")
print("================================")

print(f"Total UPENN       : {len(upenn_hashes)}")
print(f"Contaminated      : {len(contaminated)}")
print(f"Clean UPENN cases : {len(clean_cases)}")

# =====================================================
# SHOW FIRST 20
# =====================================================

print("\nFirst 20 clean cases:\n")

for case in clean_cases[:20]:
    print(case)