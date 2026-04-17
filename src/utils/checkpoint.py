import torch


class Checkpoint:
    """
    Tracks best val loss, saves model weights, and manages early stopping.

    Usage:
        ckpt = Checkpoint(model, save_path, patience=5)

        for epoch in range(40):
            ...
            improved, stop = ckpt.update(val_loss)
            if stop:
                break
    """

    def __init__(self, model, save_path, patience=5):
        """
        Args:
            model     : nn.Module to save
            save_path : str — path to save best weights (.pth)
            patience  : int — epochs without improvement before stopping
        """
        self.model           = model
        self.save_path       = save_path
        self.patience        = patience
        self.best_val_loss   = float('inf')
        self.patience_counter = 0

    def update(self, val_loss):
        """
        Compare val_loss to best so far.

        Returns:
            improved : bool — True if this is the best epoch so far
            stop     : bool — True if early stopping should trigger
        """
        if val_loss < self.best_val_loss:
            self.best_val_loss    = val_loss
            self.patience_counter = 0
            torch.save(self.model.state_dict(), self.save_path)
            return True, False
        else:
            self.patience_counter += 1
            stop = self.patience_counter >= self.patience
            return False, stop

    @property
    def best(self):
        return self.best_val_loss

    @property
    def counter(self):
        return self.patience_counter