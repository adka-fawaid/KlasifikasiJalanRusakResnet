# trainers/train_single.py
import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
import time
import json
from pathlib import Path
from Utils.looger import CSVLogger
from Utils.dataset_loader import make_dataloaders, save_augmented_samples
from Models.resnet50_builder import build_resnet50
from Utils.metrics import accuracy_from_logits
import os

def train_single(cfg):
    device = torch.device(cfg.get("device", "cuda" if torch.cuda.is_available() else "cpu"))
    batch_size = cfg["batch_size"]
    input_size = cfg["input_size"]
    lr = cfg["learning_rate"]
    optimizer_name = cfg["optimizer"]
    activation = cfg["activation"]
    dense_layers = cfg["dense_layers"]
    epochs = cfg["epochs"]
    freeze_backbone = cfg.get("freeze_backbone", True)
    use_amp = cfg.get("use_amp", True)
    num_workers = cfg.get("num_workers", 4)
    pin_memory = cfg.get("pin_memory", True)
    out_dir = Path(cfg.get("output_dir", "outputs"))
    out_dir.mkdir(parents=True, exist_ok=True)

    # dataloaders
    train_loader, val_loader, test_loader, train_ds, val_ds, test_ds = make_dataloaders(
        cfg["data_proc_dir"], input_size=input_size, batch_size=batch_size, num_workers=num_workers, pin_memory=pin_memory)

    # save some augmented samples for inspection (only once)
    save_augmented_samples(Path(cfg["data_proc_dir"]) / "train", input_size, out_dir / "aug_samples", per_class=cfg.get("save_aug_samples_per_class", 10))

    # model
    model = build_resnet50(num_classes=cfg["num_classes"],
                           dense_layers=dense_layers,
                           activation=activation,
                           pretrained=True,
                           freeze_backbone=freeze_backbone)
    model = model.to(device)

    # loss
    criterion = nn.CrossEntropyLoss()

    # optimizer
    params = filter(lambda p: p.requires_grad, model.parameters())
    if optimizer_name.lower() == "sgd":
        optimizer = optim.SGD(params, lr=lr, momentum=0.9)
    elif optimizer_name.lower() == "adamw":
        optimizer = optim.AdamW(params, lr=lr)
    else:
        optimizer = optim.Adam(params, lr=lr)

    scaler = torch.cuda.amp.GradScaler() if use_amp and device.type == "cuda" else None

    logger = CSVLogger(out_dir / "results_csv")
    run_params = {k:v for k,v in cfg.items() if k in ["optimizer","activation","batch_size","learning_rate","dense_layers","input_size","epochs","freeze_backbone"]}

    epoch_logs = []
    for epoch in range(1, epochs+1):
        t0 = time.time()

        # train
        model.train()
        train_loss = 0.0
        train_correct = 0
        train_total = 0
        pbar = tqdm(train_loader, desc=f"Epoch {epoch}/{epochs} - Train")
        for images, targets in pbar:
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
            pbar.set_postfix({"loss": f"{train_loss/train_total:.4f}", "acc": f"{train_correct/train_total:.4f}"})

        train_loss = train_loss / train_total
        train_acc = train_correct / train_total

        # validate
        model.eval()
        val_loss = 0.0
        val_correct = 0
        val_total = 0
        with torch.no_grad():
            for images, targets in tqdm(val_loader, desc="Validate"):
                images, targets = images.to(device), targets.to(device)
                with torch.cuda.amp.autocast(enabled=(scaler is not None)):
                    outputs = model(images)
                    loss = criterion(outputs, targets)
                val_loss += loss.item() * images.size(0)
                c, preds = accuracy_from_logits(outputs, targets)
                val_correct += c
                val_total += images.size(0)

        val_loss = val_loss / val_total
        val_acc = val_correct / val_total
        t1 = time.time()

        epoch_log = {
            "epoch": epoch,
            "train_loss": train_loss,
            "val_loss": val_loss,
            "train_acc": train_acc,
            "val_acc": val_acc,
            "time_epoch_s": t1 - t0
        }
        epoch_logs.append(epoch_log)

        print(f"[Epoch {epoch}] train_loss={train_loss:.4f} val_loss={val_loss:.4f} train_acc={train_acc:.4f} val_acc={val_acc:.4f} time={t1-t0:.1f}s")

        # checkpoint
        if epoch % cfg.get("checkpoint_every", 1) == 0:
            ckpt_dir = out_dir / "checkpoints"
            ckpt_dir.mkdir(parents=True, exist_ok=True)
            torch.save({
                "epoch": epoch,
                "model_state": model.state_dict(),
                "optimizer_state": optimizer.state_dict(),
                "cfg": cfg
            }, ckpt_dir / f"ckpt_epoch_{epoch}.pth")

    # save CSV log
    csv_path = logger.log_run(run_params, epoch_logs)
    print("Saved run CSV:", csv_path)

    # After training: save some prediction examples (correct + incorrect) up to per-class limit
    save_prediction_samples(model, test_loader, cfg, out_dir / "preds_samples", device, max_per_class=cfg.get("save_pred_samples_per_class", 10))

def save_prediction_samples(model, dataloader, cfg, out_dir, device, max_per_class=10):
    from torchvision.transforms.functional import to_pil_image
    import os
    import torch
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    class_counters = {i:0 for i in range(cfg["num_classes"])}
    wrong_counters = {i:0 for i in range(cfg["num_classes"])}
    class_names = cfg["class_names"]
    model.eval()
    with torch.no_grad():
        for images, targets in dataloader:
            images, targets = images.to(device), targets.to(device)
            outputs = model(images)
            preds = torch.argmax(outputs, dim=1)
            for i in range(images.size(0)):
                lbl = int(targets[i].cpu().item())
                pred = int(preds[i].cpu().item())
                # save correct examples up to max_per_class
                if lbl == pred and class_counters[lbl] < max_per_class:
                    save_img_tensor(images[i].cpu(), out_dir / "correct" / class_names[lbl], f"correct_{class_names[lbl]}_{class_counters[lbl]}.png")
                    class_counters[lbl] += 1
                # save misclassified up to max_per_class
                if lbl != pred and wrong_counters[lbl] < max_per_class:
                    save_img_tensor(images[i].cpu(), out_dir / "misclassified" / class_names[lbl], f"mis_{class_names[lbl]}_pred_{class_names[pred]}_{wrong_counters[lbl]}.png")
                    wrong_counters[lbl] += 1
            # break early if all saved
            if all(class_counters[c] >= max_per_class and wrong_counters[c] >= max_per_class for c in range(cfg["num_classes"])):
                break

def save_img_tensor(tensor, out_dir, name):
    from torchvision.transforms.functional import to_pil_image
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    # unnormalize
    mean = torch.tensor([0.485,0.456,0.406]).view(3,1,1)
    std = torch.tensor([0.229,0.224,0.225]).view(3,1,1)
    t = tensor * std + mean
    img = to_pil_image(t.clamp(0,1))
    img.save(out_dir / name)
