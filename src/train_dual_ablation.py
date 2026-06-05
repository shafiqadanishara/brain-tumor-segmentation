"""
train_dual_ablation.py
======================
Script ablasi untuk mendiagnosis mengapa train loss > val loss pada fold 0.

Tambahan flag dibanding train_dual.py:
  --fold        : pilih fold mana yang dilatih (default: 0)
  --no_aug      : matikan SEMUA augmentasi pada dataset train
  --no_elastic  : matikan hanya elastic deformation, augmentasi lain tetap jalan

Contoh penggunaan:
  # Eksperimen 1 — tanpa augmentasi sama sekali (fold 0)
  python train_dual_ablation.py --modality t1ce_flair --fold 0 --no_aug

  # Eksperimen 2 — tanpa elastic deform saja (fold 0)
  python train_dual_ablation.py --modality t1ce_flair --fold 0 --no_elastic

  # Normal (sama seperti train_dual.py, sebagai baseline)
  python train_dual_ablation.py --modality t1ce_flair --fold 0
"""

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
}


# ----------------------------------------
# HELPER: Resolve augmentation mode
# ----------------------------------------
def resolve_aug_mode(no_aug: bool, no_elastic: bool):
    """
    Returns (augment_flag, elastic_flag, tag_suffix) based on CLI flags.

    augment_flag : bool — apakah BraTSDataset3D dipanggil dengan augment=True
    elastic_flag : bool — apakah elastic deformation diaktifkan
                          (hanya relevan jika augment_flag=True)
    tag_suffix   : str  — ditambahkan ke nama fold agar checkpoint tidak
                          bertabrakan dengan run normal
    """
    if no_aug:
        return False, False, "_noaug"
    elif no_elastic:
        return True, False, "_noelastic"
    else:
        return True, True, ""


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
def train_fold(fold_idx, train_cases, val_cases, args, channels, device,
               augment_flag: bool, elastic_flag: bool, tag_suffix: str):

    aug_desc = (
        "NO augmentation" if not augment_flag
        else ("aug WITHOUT elastic" if not elastic_flag
              else "FULL augmentation")
    )

    print(f"\n{'='*60}")
    print(f"  FOLD {fold_idx}  |  modality: {args.modality}")
    print(f"  Augmentation mode : {aug_desc}")
    print(f"  train={len(train_cases)} cases  |  val={len(val_cases)} cases")
    print(f"{'='*60}")

    # ---- Datasets ----
    data_root = "data/raw/ASNR-MICCAI-BraTS2023-GLI-Challenge-TrainingData"

    # BraTSDataset3D sudah support elastic= di konstruktor, langsung pakai
    full_train = BraTSDataset3D(data_root, augment=augment_flag, elastic=elastic_flag)
    full_val   = BraTSDataset3D(data_root, augment=False)

    all_cases = full_train.cases

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
    model     = UNet3D(in_channels=len(channels), out_channels=3).to(device)
    optimizer = optim.Adam(model.parameters(), lr=args.lr)

    # ---- Paths — pakai tag_suffix agar tidak menimpa checkpoint normal ----
    fold_tag     = f"fold{fold_idx}_{args.modality}{tag_suffix}"
    out_base     = f"experiments/dual/output/ablation"
    save_path    = f"{out_base}/checkpoints/best_{fold_tag}.pth"
    history_path = f"{out_base}/history/history_{fold_tag}.json"
    csv_path     = f"{out_base}/history/history_{fold_tag}.csv"
    resume_path  = f"{out_base}/checkpoints/resume_{fold_tag}_latest.pth"

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
        print(f"\n[Fold {fold_idx} | {aug_desc}] Epoch {epoch + 1}/{args.epochs}")

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

    augment_flag, elastic_flag, tag_suffix = resolve_aug_mode(args.no_aug, args.no_elastic)

    print(f"Device      : {device}")
    print(f"Modality    : {args.modality}")
    print(f"Fold target : {args.fold}")
    print(f"Augment     : {augment_flag}  |  Elastic: {elastic_flag}")
    print(f"Tag suffix  : '{tag_suffix}' (kosong = run normal)")

    channels = MODALITY_CHANNELS[args.modality]

    with open(args.folds_json, "r") as f:
        folds = json.load(f)

    # Cari fold yang diminta
    target_fold = next((fd for fd in folds if fd["fold"] == args.fold), None)
    if target_fold is None:
        raise ValueError(
            f"Fold {args.fold} tidak ditemukan di {args.folds_json}. "
            f"Fold yang tersedia: {[fd['fold'] for fd in folds]}"
        )

    best_loss = train_fold(
        fold_idx    = target_fold["fold"],
        train_cases = target_fold["train"],
        val_cases   = target_fold["val"],
        args        = args,
        channels    = channels,
        device      = device,
        augment_flag= augment_flag,
        elastic_flag= elastic_flag,
        tag_suffix  = tag_suffix,
    )

    print(f"\nBest val loss (fold {args.fold}, {tag_suffix or 'normal'}): {best_loss:.4f}")


# ----------------------------------------
# ARGPARSE
# ----------------------------------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Ablation training — kontrol augmentasi per fold"
    )

    parser.add_argument(
        "--modality", type=str, required=True,
        choices=list(MODALITY_CHANNELS.keys()),
        help="Pasangan modalitas yang digunakan"
    )
    parser.add_argument(
        "--fold", type=int, default=0,
        help="Nomor fold yang dilatih (default: 0)"
    )
    parser.add_argument(
        "--folds_json", type=str, default="data/folds/folds.json",
        help="Path ke folds.json dari splitDataset.py"
    )

    # --- Augmentation control ---
    aug_group = parser.add_mutually_exclusive_group()
    aug_group.add_argument(
        "--no_aug", action="store_true",
        help="Matikan SEMUA augmentasi pada set training"
    )
    aug_group.add_argument(
        "--no_elastic", action="store_true",
        help="Matikan hanya elastic deformation; augmentasi lain tetap aktif"
    )

    # --- Hyperparameters ---
    parser.add_argument("--epochs",      type=int,   default=100)
    parser.add_argument("--batch_size",  type=int,   default=4)
    parser.add_argument("--lr",          type=float, default=1e-4)
    parser.add_argument("--patience",    type=int,   default=10)
    parser.add_argument("--num_workers", type=int,   default=2)

    args = parser.parse_args()

    # Validasi: --no_aug dan --no_elastic tidak bisa bersamaan (sudah ditangani
    # oleh mutually_exclusive_group, tapi kita tambah pesan yang jelas)
    if args.no_aug and args.no_elastic:
        parser.error("--no_aug dan --no_elastic tidak bisa digunakan bersamaan.")

    main(args)