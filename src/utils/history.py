import json

REGIONS = ["WT", "TC", "ET", "mean"]
METRICS = ["dsc", "hd95", "prec", "sens", "spec", "acc"]


def init_history(modality):
    history = {
        "modality": modality,
        "epochs": [],
        "train_loss": [],
        "val_loss": [],
    }

    for metric in METRICS:
        for region in REGIONS:
            history[f"train_{metric}_{region}"] = []
            history[f"val_{metric}_{region}"] = []

    return history


def update_history(history, epoch, train_loss, val_loss, train_m, val_m):
    history["epochs"].append(epoch + 1)
    history["train_loss"].append(train_loss)
    history["val_loss"].append(val_loss)

    for metric in METRICS:
        for region in REGIONS:
            history[f"train_{metric}_{region}"].append(train_m[f"{metric}_{region}"])
            history[f"val_{metric}_{region}"].append(val_m[f"{metric}_{region}"])

    return history


def save_history(history, path):
    with open(path, "w") as f:
        json.dump(history, f, indent=2)


def load_history(path):
    with open(path, "r") as f:
        return json.load(f)