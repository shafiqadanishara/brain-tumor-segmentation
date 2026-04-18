# src/train_ensemble.py

import argparse
from datetime import datetime
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


def run_epoch(loader, model, optimizer, device, training=True):
    model.train() if training else model.eval()

    total_loss = 0.0
    running_metrics = {}
    n_batches = 0

    context = torch.enable_grad() if training else torch.no_grad()
    phase = "Train" if training else "Val"

    with context:
        for x1, x2, mask in tqdm(loader, desc=phase, leave=False):
            x1 = x1.to(device, non_blocking=True)
            x2 = x2.to(device, non_blocking=True)
            mask = mask.to(device, non_blocking=True)

            if x1.shape[2:] != TARGET_SIZE:
                x1 = F.interpolate(
                    x1,
                    size=TARGET_SIZE,
                    mode="trilinear",
                    align_corners=False
                )

            if x2.shape[2:] != TARGET_SIZE:
                x2 = F.interpolate(
                    x2,
                    size=TARGET_SIZE,
                    mode="trilinear",
                    align_corners=False
                )

            if mask.shape[2:] != TARGET_SIZE:
                mask = F.interpolate(
                    mask,
                    size=TARGET_SIZE,
                    mode="nearest"
                )

            if training:
                optimizer.zero_grad()

            logits = model(x1, x2)
            loss = bce_dice_loss(logits, mask)

            if training:
                loss.backward()
                optimizer.step()

            total_loss += loss.item()

            batch_metrics = compute_metrics(logits.detach(), mask)
            running_metrics = accumulate_metrics(
                running_metrics,
                batch_metrics
            )

            n_batches += 1

    avg_loss = total_loss / n_batches
    avg_metrics = average_metrics(running_metrics, n_batches)

    return avg_loss, avg_metrics


def main(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # -----------------------------
    # DATASET
    # -----------------------------
    train_dataset = BraTSDualEnsembleDataset(
        "data/split/train",
        augment=True
    )

    val_dataset = BraTSDualEnsembleDataset(
        "data/split/val",
        augment=False
    )

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

    print(f"Train cases: {len(train_dataset)}")
    print(f"Val cases  : {len(val_dataset)}")

    # -----------------------------
    # LOAD PRETRAINED DUAL MODELS
    # -----------------------------
    model_a = UNet3D(in_channels=2, out_channels=3).to(device)
    model_b = UNet3D(in_channels=2, out_channels=3).to(device)

    model_a.load_state_dict(torch.load(args.model_a, map_location=device))
    model_b.load_state_dict(torch.load(args.model_b, map_location=device))

    model = DualEnsemble(model_a, model_b, out_channels=3).to(device)

    optimizer = optim.Adam(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=args.lr
    )

    # -----------------------------
    # PATHS
    # -----------------------------
    run_id = datetime.now().strftime("%Y%m%d_%H%M%S")

    save_path = f"experiments/ensemble/output/checkpoints/model_ensemble_{run_id}.pth"
    history_path = f"experiments/ensemble/output/history/history_ensemble.json"
    csv_path = f"experiments/ensemble/output/history/history_ensemble.csv"
    resume_path = f"experiments/ensemble/output/checkpoints/resume_ensemble_latest.pth"

    ckpt = Checkpoint(
        model=model,
        save_path=save_path,
        patience=args.patience,
        optimizer=optimizer,
        resume_path=resume_path
    )

    start_epoch, loaded_history = ckpt.load_resume()

    if loaded_history is not None:
        history = loaded_history
    else:
        history = init_history("ensemble")

    # -----------------------------
    # TRAIN LOOP
    # -----------------------------
    for epoch in range(start_epoch, args.epochs):
        print(f"\nStarting Epoch {epoch + 1}/{args.epochs}")

        train_loss, train_m = run_epoch(
            train_loader,
            model,
            optimizer,
            device,
            training=True
        )

        val_loss, val_m = run_epoch(
            val_loader,
            model,
            optimizer,
            device,
            training=False
        )

        log_epoch(
            epoch,
            args.epochs,
            train_loss,
            val_loss,
            train_m,
            val_m
        )

        history = update_history(
            history,
            epoch,
            train_loss,
            val_loss,
            train_m,
            val_m
        )

        save_history(history, history_path)
        save_history_csv(history, csv_path)

        ckpt.save_resume(epoch, history)

        improved, stop = ckpt.update(val_loss)

        log_checkpoint(
            improved,
            val_loss,
            ckpt.counter,
            ckpt.patience
        )

        if stop:
            log_early_stop(epoch)
            break

    plot_all(history_path)

    log_done(
        ckpt.best,
        save_path,
        history_path
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("--model_a", type=str, required=True,
                        help="checkpoint for t1ce_flair")

    parser.add_argument("--model_b", type=str, required=True,
                        help="checkpoint for t2_t1ce")

    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--patience", type=int, default=10)
    parser.add_argument("--num_workers", type=int, default=2)

    args = parser.parse_args()
    main(args)