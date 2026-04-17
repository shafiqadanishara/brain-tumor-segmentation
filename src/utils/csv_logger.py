import csv

def save_history_csv(history, path):
    keys = [k for k in history.keys() if k != "modality"]
    rows = len(history["epochs"])

    with open(path, "w", newline="") as f:
        writer = csv.writer(f)

        writer.writerow(keys)   # header

        for i in range(rows):
            row = [history[k][i] for k in keys]
            writer.writerow(row)