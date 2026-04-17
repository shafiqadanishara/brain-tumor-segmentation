import os
import shutil
from sklearn.model_selection import train_test_split

DATA_DIR = "/content/drive/MyDrive/Skripsi/BraTS_Data"
OUTPUT_DIR = "/content/drive/MyDrive/Skripsi/split"
RANDOM_SEED = 42

cases = sorted([
    d for d in os.listdir(DATA_DIR)
    if os.path.isdir(os.path.join(DATA_DIR, d))
])

train_cases, temp_cases = train_test_split(
    cases,
    test_size=0.30,
    random_state=RANDOM_SEED
)

val_cases, test_cases = train_test_split(
    temp_cases,
    test_size= 0.50,
    random_state=RANDOM_SEED
)

def move_cases(case_list, split_name):
    split_path = os.path.join(OUTPUT_DIR, split_name)
    os.makedirs(split_path, exist_ok=True)

    for case in case_list:
        src = os.path.join(DATA_DIR, case)
        dst = os.path.join(split_path, case)
        shutil.copytree(src, dst)

move_cases(train_cases, "train")
move_cases(val_cases, "val")
move_cases(test_cases, "test")

print("Done")