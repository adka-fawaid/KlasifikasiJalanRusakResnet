# Utils/dataset_loader.py
from torchvision import transforms
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader, Subset
from pathlib import Path
from PIL import Image
import torch
import random

MEAN = [0.485, 0.456, 0.406]
STD  = [0.229, 0.224, 0.225]

def get_transforms(input_size, is_train=True):
    if is_train:
        return transforms.Compose([
            transforms.RandomRotation(15),
            transforms.RandomResizedCrop(input_size, scale=(0.9, 1.0)),
            transforms.ColorJitter(brightness=0.2, contrast=0.2),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.ToTensor(),
            transforms.Normalize(MEAN, STD)
        ])
    else:
        return transforms.Compose([
            transforms.Resize((input_size, input_size)),
            transforms.ToTensor(),
            transforms.Normalize(MEAN, STD)
        ])

def make_dataloaders(data_dir, input_size=128, batch_size=8, num_workers=4, pin_memory=True):
    data_dir = Path(data_dir)
    train_dir = data_dir / "train"
    val_dir = data_dir / "val"
    test_dir = data_dir / "test"

    train_ds = ImageFolder(str(train_dir), transform=get_transforms(input_size, is_train=True))
    val_ds = ImageFolder(str(val_dir), transform=get_transforms(input_size, is_train=False))
    test_ds = ImageFolder(str(test_dir), transform=get_transforms(input_size, is_train=False))

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=num_workers, pin_memory=pin_memory)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=pin_memory)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=pin_memory)

    return train_loader, val_loader, test_loader, train_ds, val_ds, test_ds

def save_augmented_samples_once(train_folder, input_size, out_dir="outputs/aug_samples", per_class=10, seed=42):
    import torchvision.transforms.functional as TF
    from torchvision.transforms import ToPILImage
    train_folder = Path(train_folder)
    if not train_folder.exists():
        return
    transform = get_transforms(input_size, is_train=True)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    classes = [d.name for d in train_folder.iterdir() if d.is_dir()]
    for cls in classes:
        imgs = list((train_folder / cls).glob("*"))
        random.shuffle(imgs)
        saved = 0
        cls_out = out_dir / cls
        cls_out.mkdir(parents=True, exist_ok=True)
        for p in imgs:
            if saved >= per_class:
                break
            try:
                img = Image.open(p).convert("RGB")
                t_img = transform(img)
                t_img = t_img * torch.tensor(STD).view(3,1,1) + torch.tensor(MEAN).view(3,1,1)
                pil = ToPILImage()(t_img.clamp(0,1))
                pil.save(cls_out / f"{p.stem}_aug_{saved}.png")
                saved += 1
            except Exception as e:
                print("skip", p, e)
                continue
