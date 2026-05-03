import argparse
import json
import os
import torch
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset
from tqdm import tqdm

from src.dataset.dataset3D import BraTSDataset3D
from src.models.unet import UNet3D
from src.losses.combinedLoss import bce_dice_loss

from src.utils.csv_logger import save_history_csv
from src.utils.metrics import (
    compute_metrics,
    accumulate_metrics,
    average_metrics
)

from src.utils.checkpoint import Checkpoint
from src.utils.history import (
    init_history,
    update_history,
    save_history
)

from src.utils.logger import (
    log_epoch,
    log_checkpoint,
    log_early_stop,
    log_done
)

from src.utils.plot import plot_all


TARGET_SIZE = (128, 128, 128)

MODALITY_CHANNELS = {
    "t2_t1ce":    [2, 1],
    "t1ce_flair": [1, 3],
    "t2_flair":   [2, 3],
    # "t1_t1ce":    [0, 1],
    # "t1_flair":   [0, 3],
    # "t1_t2":      [0, 2],
}


# ----------------------------------------
# ONE EPOCH
# ----------------------------------------
def run_epoch(loader, model, optimizer, channels, device, training=True):
    model.train() if training else model.eval()

    total_loss = 0.0
    running_metrics = {}
    n_batches = 0

    context = torch.enable_grad() if training else torch.no_grad()
    phase = "Train" if training else "Val"

    with context:
        for img, mask, _ in tqdm(loader, desc=phase, leave=False):
            img  = img.to(device, non_blocking=True)
            mask = mask.to(device, non_blocking=True)

            img = img[:, channels, :, :, :]

            if img.shape[2:] != TARGET_SIZE:
                img = F.interpolate(
                    img, size=TARGET_SIZE,
                    mode="trilinear", align_corners=False
                )

            if mask.shape[2:] != TARGET_SIZE:
                mask = F.interpolate(mask, size=TARGET_SIZE, mode="nearest")

            if training:
                optimizer.zero_grad()

            logits = model(img)
            loss   = bce_dice_loss(logits, mask)

            if training:
                loss.backward()
                optimizer.step()

            total_loss += loss.item()

            batch_metrics   = compute_metrics(logits.detach(), mask)
            running_metrics = accumulate_metrics(running_metrics, batch_metrics)
            n_batches += 1

    avg_loss    = total_loss / n_batches
    avg_metrics = average_metrics(running_metrics, n_batches)

    return avg_loss, avg_metrics


# ----------------------------------------
# SINGLE-FOLD TRAINING
# ----------------------------------------
def train_fold(fold_idx, train_cases, val_cases, args, channels, device):
    print(f"\n{'='*60}")
    print(f"  FOLD {fold_idx}  |  modality: {args.modality}")
    print(f"  train={len(train_cases)} cases  |  val={len(val_cases)} cases")
    print(f"{'='*60}")

    # ---- Datasets ----
    # Full dataset (augment=True); Subset by case list
    full_train = BraTSDataset3D("data/raw", augment=True)
    full_val   = BraTSDataset3D("data/raw", augment=False)

    # Map case names to indices
    all_cases = full_train.cases  # sorted list

    train_idx = [all_cases.index(c) for c in train_cases]
    val_idx   = [all_cases.index(c) for c in val_cases]

    train_dataset = Subset(full_train, train_idx)
    val_dataset   = Subset(full_val,   val_idx)

    pin = torch.cuda.is_available()

    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=pin
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=pin
    )

    # ---- Model ----
    model = UNet3D(in_channels=len(channels), out_channels=3).to(device)
    optimizer = optim.Adam(model.parameters(), lr=args.lr)

    # ---- Paths ----
    fold_tag    = f"fold{fold_idx}_{args.modality}"
    save_path   = f"experiments/dual/output/checkpoints/best_{fold_tag}.pth"
    history_path= f"experiments/dual/output/history/history_{fold_tag}.json"
    csv_path    = f"experiments/dual/output/history/history_{fold_tag}.csv"
    resume_path = f"experiments/dual/output/checkpoints/resume_{fold_tag}_latest.pth"

    os.makedirs(os.path.dirname(save_path),    exist_ok=True)
    os.makedirs(os.path.dirname(history_path), exist_ok=True)

    # ---- Checkpoint ----
    ckpt = Checkpoint(
        model=model,
        save_path=save_path,
        patience=args.patience,
        optimizer=optimizer,
        resume_path=resume_path
    )

    start_epoch, loaded_history = ckpt.load_resume()
    history = loaded_history if loaded_history is not None else init_history(fold_tag)

    # ---- Train loop ----
    for epoch in range(start_epoch, args.epochs):
        print(f"\n[Fold {fold_idx}] Epoch {epoch + 1}/{args.epochs}")

        train_loss, train_m = run_epoch(
            train_loader, model, optimizer, channels, device, training=True
        )
        val_loss, val_m = run_epoch(
            val_loader, model, optimizer, channels, device, training=False
        )

        log_epoch(epoch, args.epochs, train_loss, val_loss, train_m, val_m)

        history = update_history(history, epoch, train_loss, val_loss, train_m, val_m)
        save_history(history, history_path)
        save_history_csv(history, csv_path)

        ckpt.save_resume(epoch, history)
        improved, stop = ckpt.update(val_loss)
        log_checkpoint(improved, val_loss, ckpt.counter, ckpt.patience)

        if stop:
            log_early_stop(epoch)
            break

    plot_all(history_path)
    log_done(ckpt.best, save_path, history_path)

    return ckpt.best


# ----------------------------------------
# MAIN
# ----------------------------------------
def main(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print(f"Device   : {device}")
    print(f"Modality : {args.modality}")

    channels = MODALITY_CHANNELS[args.modality]

    # Load fold definitions generated by splitDataset.py
    with open(args.folds_json, "r") as f:
        folds = json.load(f)

    fold_best_losses = []

    for fold in folds:
        fold_idx    = fold["fold"]
        train_cases = fold["train"]
        val_cases   = fold["val"]

        best_loss = train_fold(
            fold_idx, train_cases, val_cases,
            args, channels, device
        )

        fold_best_losses.append(best_loss)
        print(f"\n[Fold {fold_idx}] Best val loss: {best_loss:.4f}")

    print("\n" + "="*60)
    print("3-FOLD CROSS-VALIDATION SUMMARY")
    print("="*60)
    for i, loss in enumerate(fold_best_losses):
        print(f"  Fold {i}: best val loss = {loss:.4f}")
    print(f"  Mean  : {sum(fold_best_losses)/len(fold_best_losses):.4f}")
    print("="*60)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--modality",
        type=str,
        required=True,
        choices=list(MODALITY_CHANNELS.keys())
    )
    parser.add_argument(
        "--folds_json",
        type=str,
        default="data/folds/folds.json",
        help="Path to folds.json generated by splitDataset.py"
    )
    parser.add_argument("--epochs",      type=int,   default=100)
    parser.add_argument("--batch_size",  type=int,   default=4)
    parser.add_argument("--lr",          type=float, default=1e-4)
    parser.add_argument("--patience",    type=int,   default=10)
    parser.add_argument("--num_workers", type=int,   default=2)

    args = parser.parse_args()
    main(args)