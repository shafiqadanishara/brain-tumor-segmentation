def log_epoch(epoch, total_epochs, train_loss, val_loss, train_m, val_m):
    print(f"\nEpoch {epoch + 1}/{total_epochs}")
    print("-" * 60)
    print(f"{'Metric':<15}{'Train':>12}{'Val':>12}")

    rows = [
        ("Loss", train_loss, val_loss),

        ("DSC Mean", train_m["dsc_mean"], val_m["dsc_mean"]),
        ("DSC WT", train_m["dsc_WT"], val_m["dsc_WT"]),
        ("DSC TC", train_m["dsc_TC"], val_m["dsc_TC"]),
        ("DSC ET", train_m["dsc_ET"], val_m["dsc_ET"]),

        ("HD95 Mean", train_m["hd95_mean"], val_m["hd95_mean"]),
        ("HD95 WT", train_m["hd95_WT"], val_m["hd95_WT"]),
        ("HD95 TC", train_m["hd95_TC"], val_m["hd95_TC"]),
        ("HD95 ET", train_m["hd95_ET"], val_m["hd95_ET"]),

        ("Prec Mean", train_m["prec_mean"], val_m["prec_mean"]),
        ("Sens Mean", train_m["sens_mean"], val_m["sens_mean"]),
        ("Spec Mean", train_m["spec_mean"], val_m["spec_mean"]),
        ("Acc Mean", train_m["acc_mean"], val_m["acc_mean"]),
    ]

    for name, tr, va in rows:
        print(f"{name:<15}{tr:>12.4f}{va:>12.4f}")


def log_checkpoint(improved, val_loss, patience_counter, patience):
    if improved:
        print(f"\nBest model saved (val loss: {val_loss:.4f})")
    else:
        print(f"\nNo improvement ({patience_counter}/{patience})")


def log_early_stop(epoch):
    print(f"\nEarly stopping at epoch {epoch + 1}")


def log_done(best_val_loss, save_path, history_path):
    print("\nTraining Finished")
    print(f"Best Val Loss : {best_val_loss:.4f}")
    print(f"Model Saved   : {save_path}")
    print(f"History Saved : {history_path}")