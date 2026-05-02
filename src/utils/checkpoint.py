import os
import torch


class Checkpoint:
    """
    Best-model checkpoint + resume checkpoint + early stopping
    """

    def __init__(self, model, save_path, patience=5, optimizer=None, resume_path=None):
        self.model = model
        self.save_path = save_path
        self.patience = patience
        self.optimizer = optimizer
        self.resume_path = resume_path

        self.best_val_loss = float("inf")
        self.patience_counter = 0

    def update(self, val_loss):
        """
        Save best model if improved.
        """
        if val_loss < self.best_val_loss:
            self.best_val_loss = val_loss
            self.patience_counter = 0

            # Best model for testing
            torch.save(self.model.state_dict(), self.save_path)

            return True, False

        else:
            self.patience_counter += 1
            stop = self.patience_counter >= self.patience
            return False, stop

    def save_resume(self, epoch, history):
        """
        Save latest training state every epoch.
        """
        if self.resume_path is None:
            return

        checkpoint = {
            "epoch": epoch,
            "model_state_dict": self.model.state_dict(),
            "best_val_loss": self.best_val_loss,
            "patience_counter": self.patience_counter,
            "history": history
        }

        if self.optimizer is not None:
            checkpoint["optimizer_state_dict"] = self.optimizer.state_dict()

        torch.save(checkpoint, self.resume_path)

    def load_resume(self):
        """
        Load latest checkpoint if exists.
        Returns start_epoch, history
        """
        if self.resume_path is None or not os.path.exists(self.resume_path):
            return 0, None

        checkpoint = torch.load(self.resume_path, map_location="cpu")

        self.model.load_state_dict(checkpoint["model_state_dict"])

        if self.optimizer is not None and "optimizer_state_dict" in checkpoint:
            self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])

        self.best_val_loss = checkpoint["best_val_loss"]
        self.patience_counter = checkpoint["patience_counter"]

        start_epoch = checkpoint["epoch"] + 1
        history = checkpoint["history"]

        print(f"Resumed from epoch {start_epoch}")

        return start_epoch, history

    @property
    def best(self):
        return self.best_val_loss

    @property
    def counter(self):
        return self.patience_counter