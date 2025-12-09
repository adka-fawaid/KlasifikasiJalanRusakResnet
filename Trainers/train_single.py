# Trainers/train_single.py
import time
from pathlib import Path
import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
from Models.resnet50_builder import build_resnet50
from Utils.dataset_loader import make_dataloaders, save_augmented_samples_once
from Utils.metrics import accuracy_from_logits
import numpy as np

def train_single(cfg):
    """
    Returns: list of dict rows (one dict per epoch)
    Each row contains keys:
    epoch, train_acc, val_acc, test_acc, train_loss, val_loss, epoch_time
    """
    device = torch.device("cuda" if (torch.cuda.is_available() and cfg.get("device","cuda") == "cuda") else "cpu")
    batch_size = int(cfg.get("batch_size", 8))
    input_size = int(cfg.get("input_size", 128))
    lr = float(cfg.get("learning_rate", 0.001))
    optimizer_name = cfg.get("optimizer", "Adam")
    activation = cfg.get("activation", "ReLU")
    dense_layers = int(cfg.get("dense_layers", 1))
    epochs = int(cfg.get("epochs", 10))
    freeze_backbone = bool(cfg.get("freeze_backbone", True))
    use_amp = bool(cfg.get("use_amp", True))
    num_workers = int(cfg.get("num_workers", 4))
    pin_memory = bool(cfg.get("pin_memory", True))
    out_dir = Path(cfg.get("output_dir", "outputs/run_single"))
    out_dir.mkdir(parents=True, exist_ok=True)

    # Determine data directory and augmentation settings
    use_augmented = cfg.get("use_augmented_data", False)
    if use_augmented:
        data_dir = cfg.get("data_augmented_dir", "Data_augmented")
        print(f"📂 Using pre-augmented data from: {data_dir}")
    else:
        data_dir = cfg.get("data_proc_dir", "Data_processed")
        print(f"📂 Using original data with runtime augmentation from: {data_dir}")

    # dataloaders
    train_loader, val_loader, test_loader, train_ds, val_ds, test_ds = make_dataloaders(
        data_dir,
        input_size=input_size,
        batch_size=batch_size,
        num_workers=num_workers,
        pin_memory=pin_memory,
        use_augmented_data=use_augmented
    )

    # Skip saving sample images if configured
    if cfg.get("save_sample_outputs", True):
        try:
            save_augmented_samples_once(Path(data_dir) / "train",
                                       input_size,
                                       out_dir / "aug_samples",
                                       per_class=cfg.get("save_aug_samples_per_class", 10))
        except Exception as e:
            print("Warning: could not save aug samples:", e)
    else:
        print("⏩ Skipping sample output generation (save_sample_outputs=False)")

    model = build_resnet50(num_classes=cfg.get("num_classes", 3),
                           dense_layers=dense_layers,
                           activation=activation,
                           pretrained=True,
                           freeze_backbone=freeze_backbone)
    model = model.to(device)

    criterion = nn.CrossEntropyLoss()
    params = filter(lambda p: p.requires_grad, model.parameters())
    if optimizer_name.lower() == "sgd":
        optimizer = optim.SGD(params, lr=lr, momentum=0.9)
    elif optimizer_name.lower() == "adamw":
        optimizer = optim.AdamW(params, lr=lr)
    else:
        optimizer = optim.Adam(params, lr=lr)

    scaler = torch.cuda.amp.GradScaler() if (use_amp and device.type == "cuda") else None

    epoch_rows = []
    total_time = 0.0

    for epoch in range(1, epochs + 1):
        t0 = time.time()
        model.train()
        r_loss = 0.0
        r_correct = 0
        r_total = 0

        pbar = tqdm(train_loader, desc=f"Train E{epoch}/{epochs}")
        for imgs, targets in pbar:
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
            pbar.set_postfix(loss=r_loss/(r_total+1e-12), acc=r_correct/(r_total+1e-12))

        train_loss = r_loss / (r_total + 1e-12)
        train_acc = r_correct / (r_total + 1e-12)

        # validation
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
        val_loss = v_loss / (v_total + 1e-12)
        val_acc = v_correct / (v_total + 1e-12)

        # test (per-epoch as requested)
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
        test_loss = t_loss / (t_total + 1e-12)
        test_acc = t_correct / (t_total + 1e-12)

        epoch_time = time.time() - t0
        total_time += epoch_time

        print(f"[E{epoch}] train_acc={train_acc:.4f} val_acc={val_acc:.4f} test_acc={test_acc:.4f} train_loss={train_loss:.4f} val_loss={val_loss:.4f} time={epoch_time:.1f}s")

        # save best checkpoint if improved on val
        # (optional: can save per-epoch too)
        # We'll save best on val_acc
        # (create checkpoints dir)
        ckpt_dir = out_dir / "checkpoints"
        ckpt_dir.mkdir(parents=True, exist_ok=True)
        # Save every epoch as small convenience (optional)
        torch.save({
            "epoch": epoch,
            "model_state": model.state_dict(),
            "optimizer_state": optimizer.state_dict(),
            "cfg": cfg
        }, ckpt_dir / f"epoch_{epoch}.pth")

        row = {
            "epoch": int(epoch),
            "train_acc": float(train_acc),
            "val_acc": float(val_acc),
            "test_acc": float(test_acc),
            "train_loss": float(train_loss),
            "val_loss": float(val_loss),
            "epoch_time": float(epoch_time),
            "total_time_s": float(total_time)
        }
        epoch_rows.append(row)

    return epoch_rows
