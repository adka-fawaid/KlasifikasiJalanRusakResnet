# trainers/train_kfold.py
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.model_selection import StratifiedKFold
from torchvision import transforms
import numpy as np
import json
from pathlib import Path
from tqdm import tqdm
import time
from Utils.looger import CSVLogger
from Models.resnet50_builder import build_resnet50
from Utils.metrics import accuracy_from_logits
from torchvision.datasets import ImageFolder
from torch.utils.data import Subset, DataLoader

def run_kfold(cfg):
    device = torch.device(cfg.get("device", "cuda" if torch.cuda.is_available() else "cpu"))
    k = cfg.get("k_folds", 5)
    input_size = cfg["input_size"]
    batch_size = cfg["batch_size"]
    num_workers = cfg.get("num_workers", 4)
    pin_memory = cfg.get("pin_memory", True)
    use_amp = cfg.get("use_amp", True)
    epochs = cfg["epochs"]

    dataset = ImageFolder(str(Path(cfg["data_proc_dir"]) / "train"), transform=None)
    targets = np.array(dataset.targets)
    skf = StratifiedKFold(n_splits=k, shuffle=True, random_state=cfg.get("seed",42))
    fold_idx = 0
    csv_logger = CSVLogger(Path(cfg.get("output_dir","outputs")) / "results_csv")

    for train_idx, val_idx in skf.split(np.zeros(len(targets)), targets):
        fold_idx += 1
        print(f"Fold {fold_idx}/{k}")
        # create subsets with transforms
        train_subset = Subset(dataset, train_idx)
        val_subset = Subset(dataset, val_idx)

        # assign transforms (we need to set transforms attribute since ImageFolder holds transform)
        from Utils.dataset_loader import get_transforms
        train_subset.dataset.transform = get_transforms(input_size, is_train=True)
        val_subset.dataset.transform = get_transforms(input_size, is_train=False)

        train_loader = DataLoader(train_subset, batch_size=batch_size, shuffle=True, num_workers=num_workers, pin_memory=pin_memory)
        val_loader = DataLoader(val_subset, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=pin_memory)

        # build model
        model = build_resnet50(num_classes=cfg["num_classes"],
                               dense_layers=cfg["dense_layers"],
                               activation=cfg["activation"],
                               pretrained=True,
                               freeze_backbone=cfg.get("freeze_backbone", False))
        model = model.to(device)

        params = filter(lambda p: p.requires_grad, model.parameters())
        if cfg["optimizer"].lower() == "sgd":
            optimizer = optim.SGD(params, lr=cfg["learning_rate"], momentum=0.9)
        elif cfg["optimizer"].lower() == "adamw":
            optimizer = optim.AdamW(params, lr=cfg["learning_rate"])
        else:
            optimizer = optim.Adam(params, lr=cfg["learning_rate"])

        criterion = nn.CrossEntropyLoss()
        scaler = torch.cuda.amp.GradScaler() if use_amp and device.type == "cuda" else None

        epoch_logs = []
        for epoch in range(1, epochs+1):
            t0 = time.time()
            model.train()
            train_loss = 0.0
            train_correct = 0
            train_total = 0
            for images, targets in tqdm(train_loader, desc=f"Fold {fold_idx} Epoch {epoch} Train"):
                images, targets = images.to(device), targets.to(device)
                optimizer.zero_grad()
                with torch.cuda.amp.autocast(enabled=(scaler is not None)):
                    outputs = model(images)
                    loss = criterion(outputs, targets)
                if scaler is not None:
                    scaler.scale(loss).backward()
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    loss.backward()
                    optimizer.step()
                train_loss += loss.item() * images.size(0)
                c, preds = accuracy_from_logits(outputs, targets)
                train_correct += c
                train_total += images.size(0)

            train_loss /= train_total
            train_acc = train_correct / train_total

            model.eval()
            val_loss = 0.0
            val_correct = 0
            val_total = 0
            with torch.no_grad():
                for images, targets in tqdm(val_loader, desc=f"Fold {fold_idx} Epoch {epoch} Val"):
                    images, targets = images.to(device), targets.to(device)
                    with torch.cuda.amp.autocast(enabled=(scaler is not None)):
                        outputs = model(images)
                        loss = criterion(outputs, targets)
                    val_loss += loss.item() * images.size(0)
                    c, preds = accuracy_from_logits(outputs, targets)
                    val_correct += c
                    val_total += images.size(0)
            val_loss /= val_total
            val_acc = val_correct / val_total
            t1 = time.time()

            epoch_logs.append({
                "fold": fold_idx,
                "epoch": epoch,
                "train_loss": train_loss,
                "val_loss": val_loss,
                "train_acc": train_acc,
                "val_acc": val_acc,
                "time_epoch_s": t1 - t0
            })
            print(f"[Fold {fold_idx} Epoch {epoch}] train_acc={train_acc:.4f} val_acc={val_acc:.4f}")

        # save fold CSV
        params_summary = {k:cfg[k] for k in ["optimizer","activation","batch_size","learning_rate","dense_layers","input_size","epochs"] if k in cfg}
        csv_logger.log_run({**params_summary, "fold": fold_idx}, epoch_logs)

    print("K-Fold finished.")
