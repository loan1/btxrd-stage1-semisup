import torch

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
def fp_on_normals(model, loader, device, thr=0.5):
    model.eval()
    n_normal, n_fp = 0, 0
    avg_area = 0.0
    for x, y, _ in loader:
        x = x.to(device)
        y = y.to(device)
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
        "avg_pred_area_pixels": avg_area / max(n_normal, 1),
    }

@torch.no_grad()
def sweep_threshold(model, loader, device, thrs):
    rows = []
    for t in thrs:
        met = eval_all_and_tumor_only(model, loader, device, thr=float(t))
        fp  = fp_on_normals(model, loader, device, thr=float(t))
        rows.append({"thr": float(t), **met, **fp})
    return rows

def pick_threshold(rows, max_fp_rate=0.6):
    rows_ok = [r for r in rows if r.get("dice_tumor") is not None]
    cand = [r for r in rows_ok if r["fp_rate"] <= max_fp_rate]
    if cand:
        cand.sort(key=lambda r: r["dice_tumor"], reverse=True)
        return cand[0]
    rows_ok.sort(key=lambda r: (r["dice_tumor"] * (1 - r["fp_rate"])), reverse=True)
    return rows_ok[0] if rows_ok else None
