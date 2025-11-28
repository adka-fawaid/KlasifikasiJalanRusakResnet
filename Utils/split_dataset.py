# Utils/split_dataset.py
import os
import shutil
import random
from pathlib import Path

def split_dataset(raw_dir, out_dir, train_ratio=0.7, val_ratio=0.2, test_ratio=0.1, seed=42):
    assert abs(train_ratio + val_ratio + test_ratio - 1.0) < 1e-6
    random.seed(seed)
    raw_dir = Path(raw_dir)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    classes = [d.name for d in raw_dir.iterdir() if d.is_dir()]
    if not classes:
        raise RuntimeError(f"No class folders found in {raw_dir}")

    for split in ["train", "val", "test"]:
        for cls in classes:
            (out_dir / split / cls).mkdir(parents=True, exist_ok=True)

    for cls in classes:
        imgs = list((raw_dir / cls).glob("*"))
        random.shuffle(imgs)
        n = len(imgs)
        n_train = int(n * train_ratio)
        n_val = int(n * val_ratio)
        train_imgs = imgs[:n_train]
        val_imgs = imgs[n_train:n_train+n_val]
        test_imgs = imgs[n_train+n_val:]

        for p in train_imgs:
            shutil.copy2(p, out_dir / "train" / cls / p.name)
        for p in val_imgs:
            shutil.copy2(p, out_dir / "val" / cls / p.name)
        for p in test_imgs:
            shutil.copy2(p, out_dir / "test" / cls / p.name)

    print(f"Dataset split done -> {out_dir}")
