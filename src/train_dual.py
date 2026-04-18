import argparse
from datetime import datetime
import torch
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader
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

# ----------------------------------------
# DUAL MODALITY CHANNELS
# BraTS order:
# 0=t1, 1=t1ce, 2=t2, 3=flair
# ----------------------------------------
MODALITY_CHANNELS = {
    "t2_t1ce":    [2, 1],
    "t1ce_flair": [1, 3],
    "t2_flair":   [2, 3],
    "t1_t1ce":    [0, 1],
    "t1_flair":   [0, 3],
    "t1_t2":      [0, 2],
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
        for img, mask in tqdm(loader, desc=phase, leave=False):
            img = img.to(device, non_blocking=True)
            mask = mask.to(device, non_blocking=True)

            # Select dual modality channels
            img = img[:, channels, :, :, :]

            if img.shape[2:] != TARGET_SIZE:
                img = F.interpolate(
                    img,
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

            logits = model(img)
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


# ----------------------------------------
# MAIN
# ----------------------------------------
def main(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print(f"Device   : {device}")
    print(f"Modality : {args.modality}")

    channels = MODALITY_CHANNELS[args.modality]

    # Dataset
    train_dataset = BraTSDataset3D(
        "data/split/train",
        augment=True
    )

    val_dataset = BraTSDataset3D(
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

    # Model
    model = UNet3D(
        in_channels=len(channels),   # now = 2
        out_channels=3
    ).to(device)

    optimizer = optim.Adam(
        model.parameters(),
        lr=args.lr
    )

    run_id = datetime.now().strftime("%Y%m%d_%H%M%S")

    save_path = f"experiments/dual/output/checkpoints/model_{args.modality}_{run_id}.pth"
    history_path = f"experiments/dual/output/history/history_{args.modality}.json"
    csv_path = f"experiments/dual/output/history/history_{args.modality}.csv"
    resume_path = f"experiments/dual/output/checkpoints/resume_{args.modality}_latest.pth"

    # Checkpoint manager
    ckpt = Checkpoint(
        model=model,
        save_path=save_path,
        patience=args.patience,
        optimizer=optimizer,
        resume_path=resume_path
    )

    # Resume if exists
    start_epoch, loaded_history = ckpt.load_resume()

    if loaded_history is not None:
        history = loaded_history
    else:
        history = init_history(args.modality)

    # Training loop
    for epoch in range(start_epoch, args.epochs):
        print(f"\nStarting Epoch {epoch + 1}/{args.epochs}")

        train_loss, train_m = run_epoch(
            train_loader,
            model,
            optimizer,
            channels,
            device,
            training=True
        )

        val_loss, val_m = run_epoch(
            val_loader,
            model,
            optimizer,
            channels,
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

        # Save latest checkpoint every epoch
        ckpt.save_resume(epoch, history)

        # Save best model
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

    parser.add_argument(
        "--modality",
        type=str,
        required=True,
        choices=[
            "t2_t1ce",
            "t1ce_flair",
            "t2_flair",
            "t1_t1ce",
            "t1_flair",
            "t1_t2"
        ]
    )

    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--patience", type=int, default=10)
    parser.add_argument("--num_workers", type=int, default=2)

    args = parser.parse_args()
    main(args)