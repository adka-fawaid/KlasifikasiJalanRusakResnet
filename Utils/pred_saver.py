import torch
import os
from pathlib import Path
from torchvision.transforms.functional import to_pil_image

MEAN = torch.tensor([0.485, 0.456, 0.406]).view(3,1,1)
STD  = torch.tensor([0.229, 0.224, 0.225]).view(3,1,1)

def unnormalize(t):
    """Undo ImageNet normalization: tensor -> tensor [0,1]"""
    t = t * STD + MEAN
    return torch.clamp(t, 0, 1)

def save_prediction_samples(model, test_loader, cfg, out_dir, device, max_per_class=10):
    """
    Save a few prediction samples from test set.
    Saved into:
        out_dir/correct/<class_name>/
        out_dir/incorrect/<class_name>/
    """
    out_dir = Path(out_dir)
    correct_dir = out_dir / "correct"
    incorrect_dir = out_dir / "incorrect"
    correct_dir.mkdir(parents=True, exist_ok=True)
    incorrect_dir.mkdir(parents=True, exist_ok=True)

    class_names = cfg.get("class_names", None)
    num_classes = cfg.get("num_classes", 3)

    if class_names is None:
        # fallback e.g. ["Class0", "Class1", ...]
        class_names = [f"Class{i}" for i in range(num_classes)]

    # counter per class (correct & incorrect)
    correct_count = {c: 0 for c in class_names}
    incorrect_count = {c: 0 for c in class_names}

    model.eval()
    with torch.no_grad():
        for imgs, targets in test_loader:
            imgs = imgs.to(device)
            targets = targets.to(device)
            logits = model(imgs)
            preds = torch.argmax(logits, dim=1)

            for i in range(len(imgs)):
                true_idx = targets[i].item()
                pred_idx = preds[i].item()
                true_name = class_names[true_idx]
                pred_name = class_names[pred_idx]

                # Decide folder: correct or incorrect
                if true_idx == pred_idx:
                    if correct_count[true_name] >= max_per_class:
                        continue
                    save_dir = correct_dir / true_name
                    correct_count[true_name] += 1
                else:
                    if incorrect_count[true_name] >= max_per_class:
                        continue
                    save_dir = incorrect_dir / f"{true_name}_pred_{pred_name}"
                    incorrect_count[true_name] += 1

                save_dir.mkdir(parents=True, exist_ok=True)

                img = unnormalize(imgs[i].cpu())
                pil = to_pil_image(img)
                fname = f"{true_name}_pred_{pred_name}_{correct_count[true_name]}.png"
                pil.save(save_dir / fname)

            # Stop early if all classes reached max
            done_correct = all(v >= max_per_class for v in correct_count.values())
            done_incorrect = all(v >= max_per_class for v in incorrect_count.values())
            if done_correct and done_incorrect:
                break

    print(f"📸 Saved prediction samples into: {out_dir}")
    return
