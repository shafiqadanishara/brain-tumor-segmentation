import json


REGIONS = ["WT", "TC", "ET", "mean"]
METRICS = ["dsc", "prec", "sens", "spec", "acc"]


def init_history(modality):
    """
    Initialize empty history dict.

    Args:
        modality : str — e.g. "t1ce"
    Returns:
        dict with all tracking keys pre-initialized to empty lists
    """
    history = {
        "modality":   modality,
        "epochs":     [],
        "train_loss": [],
        "val_loss":   [],
    }

    for metric in METRICS:
        for region in REGIONS:
            history[f"train_{metric}_{region}"] = []
            history[f"val_{metric}_{region}"]   = []

    return history


def update_history(history, epoch, train_loss, val_loss, train_m, val_m):
    """
    Append one epoch of results into the history dict.

    Args:
        history    : dict from init_history()
        epoch      : current epoch number (0-indexed)
        train_loss : float
        val_loss   : float
        train_m    : dict from average_metrics() — train
        val_m      : dict from average_metrics() — val
    """
    history["epochs"].append(epoch + 1)
    history["train_loss"].append(train_loss)
    history["val_loss"].append(val_loss)

    for metric in METRICS:
        for region in REGIONS:
            history[f"train_{metric}_{region}"].append(train_m[f"{metric}_{region}"])
            history[f"val_{metric}_{region}"].append(val_m[f"{metric}_{region}"])

    return history


def save_history(history, path):
    """Save history dict to JSON file."""
    with open(path, "w") as f:
        json.dump(history, f, indent=2)


def load_history(path):
    """Load history dict from JSON file."""
    with open(path, "r") as f:
        return json.load(f)