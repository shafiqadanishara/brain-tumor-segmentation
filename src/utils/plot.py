import os
import matplotlib.pyplot as plt
from src.utils.history import load_history


def _plot_curve(epochs, train_vals, val_vals, title, ylabel, save_path):
    """Helper — plot one train/val curve and save."""
    plt.figure(figsize=(8, 5))
    plt.plot(epochs, train_vals, label="Train", color="steelblue")
    plt.plot(epochs, val_vals,   label="Val",   color="tomato")
    plt.title(title)
    plt.xlabel("Epoch")
    plt.ylabel(ylabel)
    plt.legend()
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()


def plot_all(history_path, out_dir=None):
    """
    Generate all training plots from a saved history JSON.

    Plots saved:
        loss.png
        dsc.png          — DSC mean
        dsc_per_region.png  — WT / TC / ET separately
        prec.png
        sens.png
        spec.png
        acc.png

    Args:
        history_path : str — path to history_{modality}.json
        out_dir      : str — folder to save plots (default: same folder as history)
    """
    history  = load_history(history_path)
    modality = history["modality"]
    epochs   = history["epochs"]

    if out_dir is None:
        out_dir = os.path.dirname(history_path) or "."
    os.makedirs(out_dir, exist_ok=True)

    # ---- Loss ----
    _plot_curve(
        epochs,
        history["train_loss"],
        history["val_loss"],
        title=f"Loss — {modality}",
        ylabel="Loss",
        save_path=os.path.join(out_dir, f"{modality}_loss.png"),
    )

    # ---- DSC mean ----
    _plot_curve(
        epochs,
        history["train_dsc_mean"],
        history["val_dsc_mean"],
        title=f"DSC Mean — {modality}",
        ylabel="DSC",
        save_path=os.path.join(out_dir, f"{modality}_dsc_mean.png"),
    )

    # ---- DSC per region ----
    plt.figure(figsize=(8, 5))
    colors = {"WT": "steelblue", "TC": "darkorange", "ET": "green"}
    for region, color in colors.items():
        plt.plot(epochs, history[f"train_dsc_{region}"],
                 label=f"Train {region}", color=color, linestyle="--")
        plt.plot(epochs, history[f"val_dsc_{region}"],
                 label=f"Val {region}",   color=color, linestyle="-")
    plt.title(f"DSC per Region — {modality}")
    plt.xlabel("Epoch")
    plt.ylabel("DSC")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, f"{modality}_dsc_regions.png"), dpi=150)
    plt.close()

    # ---- Single-curve metrics ----
    single_metrics = {
        "prec": ("Precision Mean", "Precision"),
        "sens": ("Sensitivity Mean", "Sensitivity"),
        "spec": ("Specificity Mean", "Specificity"),
        "acc":  ("Accuracy Mean",    "Accuracy"),
    }
    for key, (title, ylabel) in single_metrics.items():
        _plot_curve(
            epochs,
            history[f"train_{key}_mean"],
            history[f"val_{key}_mean"],
            title=f"{title} — {modality}",
            ylabel=ylabel,
            save_path=os.path.join(out_dir, f"{modality}_{key}.png"),
        )

    print(f"Plots saved to: {out_dir}")
    print(f"  Files: {modality}_loss.png, {modality}_dsc_mean.png, "
          f"{modality}_dsc_regions.png, {modality}_prec.png, "
          f"{modality}_sens.png, {modality}_spec.png, {modality}_acc.png")


if __name__ == "__main__":
    import sys
    if len(sys.argv) < 2:
        print("Usage: python -m src.utils.plot <history_path> [out_dir]")
        sys.exit(1)

    history_path = sys.argv[1]
    out_dir      = sys.argv[2] if len(sys.argv) > 2 else None
    plot_all(history_path, out_dir)