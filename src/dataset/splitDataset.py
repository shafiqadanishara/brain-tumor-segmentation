import os
import json
from sklearn.model_selection import train_test_split, KFold

DATA_DIR = "data/raw/ASNR-MICCAI-BraTS2023-GLI-Challenge-TrainingData"
OUTPUT_DIR = "data/folds"
N_FOLDS = 3
TEST_SIZE = 0.15
RANDOM_SEED = 42

cases = sorted([
    d for d in os.listdir(DATA_DIR)
    if os.path.isdir(os.path.join(DATA_DIR, d))
])

# ---- Step 1: Hold out 15% as test set ----
trainval_cases, test_cases = train_test_split(
    cases,
    test_size=TEST_SIZE,
    random_state=RANDOM_SEED
)

print(f"Total cases     : {len(cases)}")
print(f"Train+Val cases : {len(trainval_cases)}")
print(f"Test cases      : {len(test_cases)}")
print()

# ---- Step 2: 3-fold CV on remaining 85% ----
kf = KFold(n_splits=N_FOLDS, shuffle=True, random_state=RANDOM_SEED)

folds = []

for fold_idx, (train_idx, val_idx) in enumerate(kf.split(trainval_cases)):
    train_cases = [trainval_cases[i] for i in train_idx]
    val_cases   = [trainval_cases[i] for i in val_idx]

    folds.append({
        "fold": fold_idx,
        "train": train_cases,
        "val":   val_cases
    })

    print(f"Fold {fold_idx}: {len(train_cases)} train | {len(val_cases)} val")

# ---- Save ----
os.makedirs(OUTPUT_DIR, exist_ok=True)

fold_json = os.path.join(OUTPUT_DIR, "folds.json")
with open(fold_json, "w") as f:
    json.dump(folds, f, indent=2)

test_json = os.path.join(OUTPUT_DIR, "test.json")
with open(test_json, "w") as f:
    json.dump(test_cases, f, indent=2)

print(f"\nFold splits saved to : {fold_json}")
print(f"Test cases saved to  : {test_json}")
print("Done")