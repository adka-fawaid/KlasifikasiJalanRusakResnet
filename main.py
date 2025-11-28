# main.py
import json
from pathlib import Path
from Trainers.train_single import train_single
from Trainers.train_kfold import run_kfold
from Utils.split_dataset import split_dataset

def main():
    cfg_path = Path("configs/config.json")
    cfg = json.load(open(cfg_path))

    # Ensure dataset split exists; ask user to run split first manually or we can do it automatically:
    data_proc_dir = Path(cfg["data_proc_dir"])
    if not data_proc_dir.exists() or not any(data_proc_dir.iterdir()):
        print("Processed data not found. Running automatic split now.")
        split_dataset(cfg["data_raw_dir"], cfg["data_proc_dir"], seed=cfg.get("seed",42))

    if cfg.get("use_kfold", False):
        run_kfold(cfg)
    else:
        train_single(cfg)

if __name__ == "__main__":
    main()
