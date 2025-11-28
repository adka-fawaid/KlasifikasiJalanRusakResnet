# Trainers/train_kfold.py
import time
from pathlib import Path
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.model_selection import StratifiedKFold
from tqdm import tqdm
from Models.resnet50_builder import build_resnet50
from Utils.dataset_loader import get_transforms, ImageFolder, Subset, DataLoader
from Utils.metrics import accuracy_from_logits
import numpy as np

def run_kfold(cfg):
    """
    Returns: list of rows per epoch per fold.
    Each row includes: fold, epoch, train_acc, val_acc, test_acc, train_loss, val_loss, epoch_time
    """
    device = torch.device("cuda" if (torch.cuda.is_available() and cfg.get("device","cuda") == "cuda") else "cpu")
    k = int(cfg.get("k_folds", 5))
    input_size = int(cfg.get("input_size", 128))
    batch_size = int(cfg.get("batch_size", 8))
    epochs = int(cfg.get("epochs", 10))
    use_amp = bool(cfg.get("use_amp", True))
    num_workers = int(cfg.get("num_workers", 4))
    pin_memory = bool(cfg.get("pin_memory", True))

    data_root = Path(cfg.get("data_proc_dir", "data_processed")) / "train"
    dataset = ImageFolder(str(data_root), transform=None)
    targets = np.array(dataset.targets)
    skf = StratifiedKFold(n_splits=k, shuffle=True, random_state=cfg.get("seed", 42))

    all_rows = []

    for fold_idx, (train_idx, val_idx) in enumerate(skf.split(np.zeros(len(targets)), targets), start=1):
        print(f"\n--- Fold {fold_idx}/{k} ---")
        train_subset = Subset(dataset, train_idx)
        val_subset = Subset(dataset, val_idx)

        train_subset.dataset.transform = get_transforms(input_size, is_train=True)
        val_subset.dataset.transform = get_transforms(input_size, is_train=False)

        train_loader = DataLoader(train_subset, batch_size=batch_size, shuffle=True, num_workers=num_workers, pin_memory=pin_memory)
        val_loader = DataLoader(val_subset, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=pin_memory)

        # rebuild model per fold
        model = build_resnet50(num_classes=cfg.get("num_classes", 3),
                               dense_layers=cfg.get("dense_layers", 1),
                               activation=cfg.get("activation", "ReLU"),
                               pretrained=True,
                               freeze_backbone=cfg.get("freeze_backbone", False))
        model = model.to(device)

        params = filter(lambda p: p.requires_grad, model.parameters())
        opt_name = cfg.get("optimizer", "Adam")
        lr = float(cfg.get("learning_rate", 0.001))
        if opt_name.lower() == "sgd":
            optimizer = optim.SGD(params, lr=lr, momentum=0.9)
        elif opt_name.lower() == "adamw":
            optimizer = optim.AdamW(params, lr=lr)
        else:
            optimizer = optim.Adam(params, lr=lr)

        criterion = nn.CrossEntropyLoss()
        scaler = torch.cuda.amp.GradScaler() if (use_amp and device.type == "cuda") else None

        total_time = 0.0

        for epoch in range(1, epochs + 1):
            t0 = time.time()
            model.train()
            r_loss = 0.0
            r_correct = 0
            r_total = 0

            for imgs, targets in tqdm(train_loader, desc=f"Fold{fold_idx} Train E{epoch}/{epochs}"):
                imgs, targets = imgs.to(device), targets.to(device)
                optimizer.zero_grad()
                with torch.cuda.amp.autocast(enabled=(scaler is not None)):
                    outputs = model(imgs)
                    loss = criterion(outputs, targets)
                if scaler is not None:
                    scaler.scale(loss).backward()
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    loss.backward()
                    optimizer.step()

                r_loss += loss.item() * imgs.size(0)
                c, preds = accuracy_from_logits(outputs, targets)
                r_correct += c
                r_total += imgs.size(0)

            train_acc = r_correct / (r_total + 1e-12)
            train_loss = r_loss / (r_total + 1e-12)

            # validate
            model.eval()
            v_loss = 0.0
            v_correct = 0
            v_total = 0
            with torch.no_grad():
                for imgs, targets in val_loader:
                    imgs, targets = imgs.to(device), targets.to(device)
                    with torch.cuda.amp.autocast(enabled=(scaler is not None)):
                        outputs = model(imgs)
                        loss = criterion(outputs, targets)
                    v_loss += loss.item() * imgs.size(0)
                    c, preds = accuracy_from_logits(outputs, targets)
                    v_correct += c
                    v_total += imgs.size(0)
            val_acc = v_correct / (v_total + 1e-12)
            val_loss = v_loss / (v_total + 1e-12)

            # test on held-out test set (per-epoch)
            # Use test folder from data_processed/test
            test_loader = DataLoader(ImageFolder(str(Path(cfg.get("data_proc_dir", "data_processed")) / "test"), transform=get_transforms(input_size, is_train=False)),
                                     batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=pin_memory)
            t_loss = 0.0
            t_correct = 0
            t_total = 0
            with torch.no_grad():
                for imgs, targets in test_loader:
                    imgs, targets = imgs.to(device), targets.to(device)
                    with torch.cuda.amp.autocast(enabled=(scaler is not None)):
                        outputs = model(imgs)
                        loss = criterion(outputs, targets)
                    t_loss += loss.item() * imgs.size(0)
                    c, preds = accuracy_from_logits(outputs, targets)
                    t_correct += c
                    t_total += imgs.size(0)
            test_acc = t_correct / (t_total + 1e-12)
            test_loss = t_loss / (t_total + 1e-12)

            epoch_time = time.time() - t0
            total_time += epoch_time

            row = {
                "fold": int(fold_idx),
                "epoch": int(epoch),
                "train_acc": float(train_acc),
                "val_acc": float(val_acc),
                "test_acc": float(test_acc),
                "train_loss": float(train_loss),
                "val_loss": float(val_loss),
                "epoch_time": float(epoch_time),
                "total_time_s": float(total_time)
            }
            all_rows.append(row)

    return all_rows
