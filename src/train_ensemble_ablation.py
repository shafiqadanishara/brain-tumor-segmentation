"""
train_ensemble_ablation.py
==========================
Script ablasi ensemble untuk fold tertentu saja.

Tambahan flag dibanding train_ensemble.py:
  --fold       : pilih fold yang dilatih (default: 2)
  --no_aug     : matikan SEMUA augmentasi
  --elastic    : aktifkan elastic deformation (default: dimatikan)

Eksperimen yang direncanakan (fold 2):
  # Baseline — sudah ada, tidak perlu dijalankan ulang
  python -m src.train_ensemble_ablation --fold 2 --model_a ... --model_b ...

  # Eksperimen 1 — tanpa augmentasi sama sekali
  python -m src.train_ensemble_ablation --fold 2 --no_aug --model_a ... --model_b ...

  # Eksperimen 2 — dengan elastic deformation
  python -m src.train_ensemble_ablation --fold 2 --elastic --model_a ... --model_b ...
"""

import argparse
import json
import os
import torch
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm

from src.models.unet import UNet3D
from src.models.dual_ensemble import DualEnsemble
from src.dataset.dataset_dual_ensemble import BraTSDualEnsembleDataset

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


# ----------------------------------------
# HELPER: build tag suffix
# ----------------------------------------
def build_tag_suffix(no_aug: bool, elastic: bool):
    """
    no_aug=True              → '_noaug'
    elastic=True             → '_elastic'
    no_aug=False, elastic=False → '' (baseline, tidak perlu tag)
    """
    if no_aug:
        return "_noaug"
    elif elastic:
        return "_elastic"
    return ""


# ----------------------------------------
# FOLD-AWARE SUBSET
# ----------------------------------------
class _FoldSubsetEnsemble(torch.utils.data.Dataset):
    def __init__(self, base_dataset, indices):
        self.base    = base_dataset
        self.indices = indices

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, i):
        return self.base[self.indices[i]]


# ----------------------------------------
# ONE EPOCH
# ----------------------------------------
def run_epoch(loader, model, optimizer, device, training=True):
    model.train() if training else model.eval()

    total_loss      = 0.0
    running_metrics = {}
    n_batches       = 0

    context = torch.enable_grad() if training else torch.no_grad()
    phase   = "Train" if training else "Val"

    with context:
        for x1, x2, mask in tqdm(loader, desc=phase, leave=False):
            x1   = x1.to(device, non_blocking=True)
            x2   = x2.to(device, non_blocking=True)
            mask = mask.to(device, non_blocking=True)

            if x1.shape[2:] != TARGET_SIZE:
                x1 = F.interpolate(x1, size=TARGET_SIZE,
                                   mode="trilinear", align_corners=False)
            if x2.shape[2:] != TARGET_SIZE:
                x2 = F.interpolate(x2, size=TARGET_SIZE,
                                   mode="trilinear", align_corners=False)
            if mask.shape[2:] != TARGET_SIZE:
                mask = F.interpolate(mask, size=TARGET_SIZE, mode="nearest")

            if training:
                optimizer.zero_grad()

            logits = model(x1, x2)
            loss   = bce_dice_loss(logits, mask)

            if training:
                loss.backward()
                optimizer.step()

            total_loss     += loss.item()
            batch_metrics   = compute_metrics(logits.detach(), mask)
            running_metrics = accumulate_metrics(running_metrics, batch_metrics)
            n_batches      += 1

    return total_loss / n_batches, average_metrics(running_metrics, n_batches)


# ----------------------------------------
# SINGLE-FOLD TRAINING
# ----------------------------------------
def train_fold(fold_idx, train_cases, val_cases, args, device,
               augment_flag: bool, elastic_flag: bool, tag_suffix: str):

    aug_desc = (
        "NO augmentation"      if not augment_flag else
        "aug WITH elastic"     if elastic_flag else
        "aug WITHOUT elastic"
    )

    print(f"\n{'='*60}")
    print(f"  FOLD {fold_idx}  |  ensemble ablation")
    print(f"  Aug mode : {aug_desc}")
    print(f"  train={len(train_cases)}  val={len(val_cases)}")
    print(f"{'='*60}")

    data_root = "data/raw/ASNR-MICCAI-BraTS2023-GLI-Challenge-TrainingData"

    full_train_ds = BraTSDualEnsembleDataset(
        data_root,
        augment=augment_flag,
        elastic=elastic_flag
    )
    full_val_ds = BraTSDualEnsembleDataset(
        data_root,
        augment=False,
        elastic=False
    )

    all_cases = full_train_ds.base.cases

    train_idx = [all_cases.index(c) for c in train_cases]
    val_idx   = [all_cases.index(c) for c in val_cases]

    train_dataset = _FoldSubsetEnsemble(full_train_ds, train_idx)
    val_dataset   = _FoldSubsetEnsemble(full_val_ds,   val_idx)

    pin = torch.cuda.is_available()
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size,
                              shuffle=True,  num_workers=args.num_workers,
                              pin_memory=pin)
    val_loader   = DataLoader(val_dataset,   batch_size=args.batch_size,
                              shuffle=False, num_workers=args.num_workers,
                              pin_memory=pin)

    # ---- Load pretrained backbones ----
    def pick_ckpt(arg_value, fold_idx):
        parts = [p.strip() for p in arg_value.split(",")]
        return parts[fold_idx] if len(parts) > 1 else parts[0]

    ckpt_a = pick_ckpt(args.model_a, fold_idx)
    ckpt_b = pick_ckpt(args.model_b, fold_idx)

    model_a = UNet3D(in_channels=2, out_channels=3).to(device)
    model_b = UNet3D(in_channels=2, out_channels=3).to(device)

    model_a.load_state_dict(torch.load(ckpt_a, map_location=device))
    model_b.load_state_dict(torch.load(ckpt_b, map_location=device))

    for p in model_a.parameters(): p.requires_grad = False
    for p in model_b.parameters(): p.requires_grad = False

    model     = DualEnsemble(model_a, model_b, out_channels=3).to(device)
    optimizer = optim.Adam(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=args.lr
    )

    # ---- Paths — ke ablation folder agar tidak menimpa ensemble normal ----
    fold_tag     = f"fold{fold_idx}{tag_suffix}"
    out_base     = "experiments/ensemble/output/ablation"
    save_path    = f"{out_base}/checkpoints/best_ensemble_{fold_tag}.pth"
    history_path = f"{out_base}/history/history_ensemble_{fold_tag}.json"
    csv_path     = f"{out_base}/history/history_ensemble_{fold_tag}.csv"
    resume_path  = f"{out_base}/checkpoints/resume_ensemble_{fold_tag}_latest.pth"

    os.makedirs(os.path.dirname(save_path),    exist_ok=True)
    os.makedirs(os.path.dirname(history_path), exist_ok=True)

    ckpt = Checkpoint(model=model, save_path=save_path, patience=args.patience,
                      optimizer=optimizer, resume_path=resume_path)

    start_epoch, loaded_history = ckpt.load_resume()
    history = loaded_history if loaded_history is not None \
              else init_history(f"ensemble_{fold_tag}")

    for epoch in range(start_epoch, args.epochs):
        print(f"\n[Fold {fold_idx} | {aug_desc}] Epoch {epoch+1}/{args.epochs}")

        train_loss, train_m = run_epoch(train_loader, model, optimizer,
                                        device, training=True)
        val_loss,   val_m   = run_epoch(val_loader,   model, optimizer,
                                        device, training=False)

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

    # Resolve flags
    augment_flag = not args.no_aug
    elastic_flag = args.elastic and not args.no_aug  # elastic tidak relevan kalau no_aug
    tag_suffix   = build_tag_suffix(args.no_aug, elastic_flag)

    print(f"Device     : {device}")
    print(f"Fold       : {args.fold}")
    print(f"Augment    : {augment_flag}  |  Elastic: {elastic_flag}")
    print(f"Tag suffix : '{tag_suffix or '(baseline)'}'")

    with open(args.folds_json) as f:
        folds = json.load(f)

    target = next((fd for fd in folds if fd["fold"] == args.fold), None)
    if target is None:
        raise ValueError(f"Fold {args.fold} tidak ditemukan di {args.folds_json}")

    best = train_fold(
        fold_idx    = target["fold"],
        train_cases = target["train"],
        val_cases   = target["val"],
        args        = args,
        device      = device,
        augment_flag= augment_flag,
        elastic_flag= elastic_flag,
        tag_suffix  = tag_suffix,
    )

    print(f"\nBest val loss: {best:.4f}  [{tag_suffix or 'baseline'}]")


# ----------------------------------------
# ARGPARSE
# ----------------------------------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Ensemble ablation — kontrol augmentasi per fold"
    )

    parser.add_argument("--model_a", type=str, required=True,
                        help="Pretrained t1ce_flair checkpoint(s), comma-separated per fold")
    parser.add_argument("--model_b", type=str, required=True,
                        help="Pretrained t2_t1ce checkpoint(s), comma-separated per fold")
    parser.add_argument("--folds_json", type=str, default="data/folds/folds.json")
    parser.add_argument("--fold",     type=int, default=2,
                        help="Fold yang dilatih (default: 2)")

    # Augmentation flags — mutually exclusive
    aug_group = parser.add_mutually_exclusive_group()
    aug_group.add_argument("--no_aug",  action="store_true",
                           help="Matikan semua augmentasi")
    aug_group.add_argument("--elastic", action="store_true",
                           help="Aktifkan elastic deformation (default: dimatikan)")

    # Hyperparams
    parser.add_argument("--epochs",      type=int,   default=100)
    parser.add_argument("--batch_size",  type=int,   default=4)
    parser.add_argument("--lr",          type=float, default=1e-4)
    parser.add_argument("--patience",    type=int,   default=10)
    parser.add_argument("--num_workers", type=int,   default=2)

    args = parser.parse_args()
    main(args)