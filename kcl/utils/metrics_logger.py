import csv
import os
import time


class MetricsLogger:
    """Append one row per epoch to ``<log_dir>/metrics.csv``.

    The file is written by rank 0 only and contains everything needed to compute the
    metrics used in the paper (final accuracy, best training loss, time to reach a target
    loss or accuracy) without any external logging service.
    """

    FIELDS = [
        "epoch", "train_loss", "test_loss", "accuracy", "lr", "epoch_time_s", "elapsed_s",
        "prediction_epoch", "mask_ratio", "predicted_num", "predict_epoch_interval",
    ]

    def __init__(self, log_dir, enabled=True, filename="metrics.csv"):
        self.enabled = enabled
        self.path = os.path.join(log_dir, filename)
        self.start_time = time.time()
        if self.enabled:
            os.makedirs(log_dir, exist_ok=True)
            with open(self.path, "w", newline="") as f:
                csv.DictWriter(f, fieldnames=self.FIELDS).writeheader()

    def log(self, **row):
        if not self.enabled:
            return
        row.setdefault("elapsed_s", round(time.time() - self.start_time, 3))
        clean = {k: ("" if row.get(k) is None else row.get(k)) for k in self.FIELDS}
        with open(self.path, "a", newline="") as f:
            csv.DictWriter(f, fieldnames=self.FIELDS).writerow(clean)
