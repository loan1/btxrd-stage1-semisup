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
def soft_dice_tumor(model, loader, device, eps=1e-6):
    model.eval()
    vals = []
    for x, y, _ in loader:
        x = x.to(device)
        y = y.to(device)
        prob = torch.sigmoid(model(x))

        gt_sum = y.sum(dim=(1,2,3))
        has_tumor = (gt_sum > 0)
        if not has_tumor.any():
            continue

        p = prob[has_tumor].view(has_tumor.sum().item(), -1)
        t = y[has_tumor].view(has_tumor.sum().item(), -1)
        inter = (p * t).sum(dim=1)
        union = p.sum(dim=1) + t.sum(dim=1)
        dice = (2*inter + eps) / (union + eps)
        vals.append(dice.mean().item())
    return sum(vals)/len(vals) if vals else 0.0

def fit_posw_resume(
    model, train_loader, val_loader, device, out_dir,
    pos_weight, lr=1e-3, epochs_total=20, thr_val=0.5, resume=True
):
    os.makedirs(out_dir, exist_ok=True)
    ckpt_last = os.path.join(out_dir, "last.pt")
    ckpt_best = os.path.join(out_dir, "best.pth")

    opt = torch.optim.Adam(model.parameters(), lr=lr)
    bce = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([float(pos_weight)], device=device))

    use_amp = str(device).startswith("cuda")
    if use_amp:
        from torch.amp import autocast, GradScaler
        scaler = GradScaler("cuda")
        amp_ctx = lambda: autocast(device_type="cuda", dtype=torch.float16)
    else:
        scaler = None
        amp_ctx = nullcontext

    start_epoch = 1
    best_soft = -1.0

    if resume and os.path.exists(ckpt_last):
        ck = torch.load(ckpt_last, map_location=device)
        model.load_state_dict(ck["model"])
        opt.load_state_dict(ck["opt"])
        if use_amp and ck.get("scaler") is not None:
            scaler.load_state_dict(ck["scaler"])
        start_epoch = ck["epoch"] + 1
        best_soft = ck.get("best_soft", best_soft)
        print(f"[RESUME] last.pt epoch {ck['epoch']} -> continue {start_epoch}")
    elif resume and os.path.exists(ckpt_best):
        model.load_state_dict(torch.load(ckpt_best, map_location=device))
        print("[RESUME] loaded best.pth (optimizer reset)")

    @torch.no_grad()
    def _bin_val_dice_tumor(model, loader, thr=0.5, eps=1e-6):
        model.eval()
        vals = []
        for x, y, _ in loader:
            x = x.to(device)
            y = y.to(device)
            prob = torch.sigmoid(model(x))
            pred = (prob > thr).float()

            gt_sum = y.sum(dim=(1,2,3))
            has_tumor = (gt_sum > 0)
            if not has_tumor.any():
                continue

            inter = (pred * y).sum(dim=(1,2,3))
            sum_pred = pred.sum(dim=(1,2,3))
            sum_gt = y.sum(dim=(1,2,3))
            dice = (2*inter + eps) / (sum_pred + sum_gt + eps)
            vals.append(dice[has_tumor].mean().item())
        return sum(vals)/len(vals) if vals else 0.0

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

        val_soft = soft_dice_tumor(model, val_loader, device)
        val_bin  = _bin_val_dice_tumor(model, val_loader, thr=thr_val)

        print(f"Epoch {ep:02d} | loss={total/len(train_loader.dataset):.4f} | "
              f"val_soft_dice_tumor={val_soft:.4f} | val_dice_tumor@{thr_val}={val_bin:.4f}")

        torch.save({
            "epoch": ep,
            "model": model.state_dict(),
            "opt": opt.state_dict(),
            "scaler": scaler.state_dict() if use_amp else None,
            "best_soft": best_soft,
        }, ckpt_last)

        if val_soft > best_soft:
            best_soft = val_soft
            torch.save(model.state_dict(), ckpt_best)

    print("Done. Best val_soft_dice_tumor:", best_soft, "best:", ckpt_best, "last:", ckpt_last)
