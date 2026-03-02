import os
import torch
import torch.nn as nn
from tqdm import tqdm
from contextlib import nullcontext

def dice_loss_with_logits(logits, target, eps=1e-6):
    prob = torch.sigmoid(logits).view(logits.size(0), -1)
    target = target.view(target.size(0), -1)
    inter = (prob * target).sum(dim=1)
    union = prob.sum(dim=1) + target.sum(dim=1)
    dice = (2*inter + eps) / (union + eps)
    return 1 - dice.mean()

@torch.no_grad()
def eval_all_and_tumor_only(model, loader, device, thr=0.5, eps=1e-6):
    model.eval()
    dice_all, iou_all = [], []
    dice_t, iou_t = [], []

    for x, y, _ in loader:
        x = x.to(device)
        y = y.to(device)

        prob = torch.sigmoid(model(x))
        pred = (prob > thr).float()

        inter = (pred * y).sum(dim=(1,2,3))
        sum_pred = pred.sum(dim=(1,2,3))
        sum_gt = y.sum(dim=(1,2,3))

        dice = (2*inter + eps) / (sum_pred + sum_gt + eps)
        iou  = (inter + eps) / (sum_pred + sum_gt - inter + eps)

        dice_all.append(dice.mean().item())
        iou_all.append(iou.mean().item())

        has_tumor = (sum_gt > 0)
        if has_tumor.any():
            dice_t.append(dice[has_tumor].mean().item())
            iou_t.append(iou[has_tumor].mean().item())

    return {
        "dice_all": sum(dice_all)/len(dice_all),
        "iou_all":  sum(iou_all)/len(iou_all),
        "dice_tumor": (sum(dice_t)/len(dice_t)) if len(dice_t)>0 else None,
        "iou_tumor":  (sum(iou_t)/len(iou_t)) if len(iou_t)>0 else None,
    }

@torch.no_grad()
def quick_stats(model, loader, device, thr=0.5):
    model.eval()
    gt_tumor = 0
    pred_tumor = 0
    n = 0
    for x, y, _ in loader:
        x = x.to(device)
        y = y.to(device)
        n += x.size(0)
        gt_sum = y.sum(dim=(1,2,3))
        gt_tumor += (gt_sum > 0).sum().item()

        prob = torch.sigmoid(model(x))
        pred = (prob > thr).float()
        pred_sum = pred.sum(dim=(1,2,3))
        pred_tumor += (pred_sum > 0).sum().item()

    return {
        "num_images": n,
        "gt_tumor_images": gt_tumor,
        "gt_normal_images": n - gt_tumor,
        "pred_tumor_images": pred_tumor,
        "pred_empty_images": n - pred_tumor,
        "gt_tumor_ratio": gt_tumor / max(n, 1)
    }

@torch.no_grad()
def fp_on_normals(model, loader, device, thr=0.5):
    model.eval()
    n_normal = 0
    n_fp = 0
    avg_area = 0.0

    for x, y, _ in loader:
        x, y = x.to(device), y.to(device)
        prob = torch.sigmoid(model(x))
        pred = (prob > thr).float()

        gt_sum = y.sum(dim=(1,2,3))
        pred_sum = pred.sum(dim=(1,2,3))

        is_normal = (gt_sum == 0)
        if is_normal.any():
            n = is_normal.sum().item()
            n_normal += n
            n_fp += (pred_sum[is_normal] > 0).sum().item()
            avg_area += pred_sum[is_normal].float().mean().item()

    return {
        "normal_images": n_normal,
        "fp_normals": n_fp,
        "fp_rate": n_fp / max(n_normal, 1),
        "avg_pred_area_pixels": avg_area / max(n_normal, 1)
    }

def fit_posw_resume(
    model, train_loader, val_loader, device, out_dir,
    pos_weight, lr=1e-3, epochs_total=20, thr_val=0.5, resume=True
):
    os.makedirs(out_dir, exist_ok=True)
    ckpt_last = os.path.join(out_dir, "last.pt")
    ckpt_best = os.path.join(out_dir, "best.pth")

    opt = torch.optim.Adam(model.parameters(), lr=lr)
    bce = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([pos_weight], device=device))

    use_amp = device.startswith("cuda")
    if use_amp:
        from torch.amp import autocast, GradScaler
        scaler = GradScaler("cuda")
        amp_ctx = lambda: autocast(device_type="cuda", dtype=torch.float16)
    else:
        scaler = None
        amp_ctx = nullcontext

    start_epoch = 1
    best_val = -1.0

    if resume and os.path.exists(ckpt_last):
        ck = torch.load(ckpt_last, map_location=device)
        model.load_state_dict(ck["model"])
        opt.load_state_dict(ck["opt"])
        if use_amp and ck.get("scaler") is not None:
            scaler.load_state_dict(ck["scaler"])
        start_epoch = ck["epoch"] + 1
        best_val = ck["best_val"]
        print(f"[RESUME] last.pt epoch {ck['epoch']} -> continue {start_epoch}")
    elif resume and os.path.exists(ckpt_best):
        model.load_state_dict(torch.load(ckpt_best, map_location=device))
        print("[RESUME] loaded best.pth (optimizer reset)")

    for ep in range(start_epoch, epochs_total + 1):
        model.train()
        total = 0.0

        for x, y, _ in tqdm(train_loader, desc=f"Train {ep}", leave=False):
            x = x.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)
            opt.zero_grad(set_to_none=True)

            with amp_ctx():
                logits = model(x)
                loss = 0.5*bce(logits, y) + 0.5*dice_loss_with_logits(logits, y)

            if use_amp:
                scaler.scale(loss).backward()
                scaler.step(opt)
                scaler.update()
            else:
                loss.backward()
                opt.step()

            total += loss.item() * x.size(0)

        val = eval_all_and_tumor_only(model, val_loader, device, thr=thr_val)
        val_dice_t = val["dice_tumor"] if val["dice_tumor"] is not None else 0.0
        print(f"Epoch {ep:02d} | loss={total/len(train_loader.dataset):.4f} | val_dice_tumor@{thr_val}={val_dice_t:.4f}")

        torch.save({
            "epoch": ep,
            "model": model.state_dict(),
            "opt": opt.state_dict(),
            "scaler": scaler.state_dict() if use_amp else None,
            "best_val": best_val,
        }, ckpt_last)

        if val_dice_t > best_val:
            best_val = val_dice_t
            torch.save(model.state_dict(), ckpt_best)

    print("Done. Best val_dice_tumor:", best_val, "best:", ckpt_best, "last:", ckpt_last)
