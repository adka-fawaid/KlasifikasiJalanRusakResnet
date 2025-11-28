# utils/logger.py
import os
import pandas as pd
from pathlib import Path
from datetime import datetime

class CSVLogger:
    def __init__(self, out_dir="outputs/results_csv"):
        self.out_dir = Path(out_dir)
        self.out_dir.mkdir(parents=True, exist_ok=True)

    def log_run(self, params: dict, epoch_logs: list):
        """
        params: dict of hyperparams
        epoch_logs: list of dicts with keys:
            epoch, train_loss, val_loss, train_acc, val_acc, time_epoch_s
        """
        run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
        rows = []
        for e in epoch_logs:
            row = {**params, **e}
            rows.append(row)
        df = pd.DataFrame(rows)
        fname = self.out_dir / f"run_{run_id}.csv"
        df.to_csv(fname, index=False)
        return fname

    def log_summary(self, summary: dict, name="summary"):
        df = pd.DataFrame([summary])
        fname = self.out_dir / f"{name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
        df.to_csv(fname, index=False)
        return fname
