import csv


def save_history_csv(history, path):
    """
    Save all history values into CSV.
    One row = one epoch
    """

    keys = [k for k in history.keys() if k != "modality"]
    rows = len(history["epochs"])

    with open(path, "w", newline="") as f:
        writer = csv.writer(f)

        # Header
        writer.writerow(keys)

        # Rows
        for i in range(rows):
            row = [history[k][i] for k in keys]
            writer.writerow(row)

    print(f"CSV saved -> {path}")