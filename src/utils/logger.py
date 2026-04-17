def log_epoch(epoch, total_epochs, train_loss, val_loss, train_m, val_m):
    """
    Print a formatted epoch summary to console.

    Args:
        epoch        : current epoch number (0-indexed)
        total_epochs : total number of epochs
        train_loss   : float
        val_loss     : float
        train_m      : dict from compute_metrics (train)
        val_m        : dict from compute_metrics (val)
    """
    print(f"\nEpoch {epoch + 1}/{total_epochs}")
    print(f"  {'Metric':<12} {'Train':>8}  {'Val':>8}")
    print(f"  {'-'*30}")
    print(f"  {'Loss':<12} {train_loss:>8.4f}  {val_loss:>8.4f}")
    print(f"  {'DSC  WT':<12} {train_m['dsc_WT']:>8.4f}  {val_m['dsc_WT']:>8.4f}")
    print(f"  {'DSC  TC':<12} {train_m['dsc_TC']:>8.4f}  {val_m['dsc_TC']:>8.4f}")
    print(f"  {'DSC  ET':<12} {train_m['dsc_ET']:>8.4f}  {val_m['dsc_ET']:>8.4f}")
    print(f"  {'DSC  Mean':<12} {train_m['dsc_mean']:>8.4f}  {val_m['dsc_mean']:>8.4f}")
    print(f"  {'Prec Mean':<12} {train_m['prec_mean']:>8.4f}  {val_m['prec_mean']:>8.4f}")
    print(f"  {'Sens Mean':<12} {train_m['sens_mean']:>8.4f}  {val_m['sens_mean']:>8.4f}")
    print(f"  {'Spec Mean':<12} {train_m['spec_mean']:>8.4f}  {val_m['spec_mean']:>8.4f}")
    print(f"  {'Acc  Mean':<12} {train_m['acc_mean']:>8.4f}  {val_m['acc_mean']:>8.4f}")


def log_checkpoint(improved, val_loss, patience_counter, patience):
    """Print checkpoint / early stopping status."""
    if improved:
        print(f"  --> Best model saved (val loss: {val_loss:.4f})")
    else:
        print(f"  --> No improvement ({patience_counter}/{patience})")


def log_early_stop(epoch):
    print(f"\nEarly stopping at epoch {epoch + 1}.")


def log_done(best_val_loss, save_path, history_path):
    print(f"\nDone. Best val loss: {best_val_loss:.4f}")
    print(f"  Model   → {save_path}")
    print(f"  History → {history_path}")