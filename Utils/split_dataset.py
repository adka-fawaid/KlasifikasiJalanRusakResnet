# utils/split_dataset.py
import os
import shutil
import random
from pathlib import Path
import json

def split_dataset(raw_dir, out_dir, train_ratio=0.7, val_ratio=0.2, test_ratio=0.1, seed=42):
    assert abs(train_ratio + val_ratio + test_ratio - 1.0) < 1e-6
    random.seed(seed)
    raw_dir = Path(raw_dir)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    for split in ["train", "val", "test"]:
        (out_dir / split).mkdir(parents=True, exist_ok=True)

    classes = [d.name for d in raw_dir.iterdir() if d.is_dir()]
    if not classes:
        raise RuntimeError(f"No class folders found in {raw_dir}")

    for cls in classes:
        imgs = list((raw_dir / cls).glob("*"))
        random.shuffle(imgs)
        n = len(imgs)
        n_train = int(n * train_ratio)
        n_val = int(n * val_ratio)
        train_imgs = imgs[:n_train]
        val_imgs = imgs[n_train:n_train+n_val]
        test_imgs = imgs[n_train+n_val:]

        for split_name, split_list in [("train", train_imgs), ("val", val_imgs), ("test", test_imgs)]:
            target_dir = out_dir / split_name / cls
            target_dir.mkdir(parents=True, exist_ok=True)
            for p in split_list:
                shutil.copy2(p, target_dir / p.name)

    print(f"Split done. Output in {out_dir}")

if __name__ == "__main__":
    cfg_path = Path("configs/config.json")
    if not cfg_path.exists():
        print("configs/config.json not found. Please create config first.")
    else:
        cfg = json.load(open(cfg_path))
        split_dataset(cfg["data_raw_dir"], cfg["data_proc_dir"], seed=cfg.get("seed", 42))
