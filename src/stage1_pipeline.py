import os
import numpy as np
import torch
from torch.utils.data import DataLoader, ConcatDataset, WeightedRandomSampler

from btxrd_dataset import BTXRDSegDataset
from unet import UNet
from data_utils import read_list, load_or_build_tumor_normal_lists, budget_split
from train_utils import fit_posw_resume
from metrics import eval_all_and_tumor_only, fp_on_normals, sweep_threshold, pick_threshold

def build_loaders(proc_dir, raw_dir, pseudo_dir, run_mode, p,
                  img_size=512, batch=16, nw=2,
                  sampling="auto_group_balance", normal_weight=1.0,
                  seed=2026, pseudo_subdir="sam_box_oracle"):
    val_ids  = read_list(f"{proc_dir}/splits/val.txt")
    test_ids = read_list(f"{proc_dir}/splits/test.txt")

    gt_masks_dir = f"{proc_dir}/masks_gt"
    train_tumor, train_normal, cache_info = load_or_build_tumor_normal_lists(
        proc_dir=proc_dir,
        raw_images_dir=f"{raw_dir}/images",
        gt_masks_dir=gt_masks_dir,
        val_ids=val_ids,
        test_ids=test_ids,
        show_progress=True,
    )

    labeled_tumor, unlabeled_tumor = budget_split(train_tumor, p=p, seed=seed)

    ds_labeled = BTXRDSegDataset(f"{raw_dir}/images", labeled_tumor, "gt", gt_masks_dir, img_size=img_size)
    ds_normal  = BTXRDSegDataset(f"{raw_dir}/images", train_normal, "gt", gt_masks_dir, img_size=img_size)

    datasets = [ds_labeled]
    if run_mode == "semi":
        ds_unlab = BTXRDSegDataset(f"{raw_dir}/images", unlabeled_tumor, "pseudo", gt_masks_dir,
                                   pseudo_masks_dir=f"{pseudo_dir}/{pseudo_subdir}", img_size=img_size)
        datasets.append(ds_unlab)
    datasets.append(ds_normal)

    sizes = [len(d) for d in datasets]
    tumor_total = sizes[0] + (sizes[1] if run_mode=="semi" else 0)
    normal_total = sizes[-1]

    if sampling == "auto_group_balance":
        normal_weight = tumor_total / max(normal_total, 1)

    weights = [1.0]*sizes[0]
    if run_mode == "semi":
        weights += [1.0]*sizes[1]
    weights += [float(normal_weight)]*sizes[-1]

    train_ds = ConcatDataset(datasets)
    sampler = WeightedRandomSampler(weights, num_samples=sum(sizes), replacement=True)
    train_loader = DataLoader(train_ds, batch_size=batch, sampler=sampler,
                              num_workers=nw, pin_memory=True, persistent_workers=(nw>0))

    val_ds  = BTXRDSegDataset(f"{raw_dir}/images", val_ids,  "gt", gt_masks_dir, img_size=img_size)
    test_ds = BTXRDSegDataset(f"{raw_dir}/images", test_ids, "gt", gt_masks_dir, img_size=img_size)
    val_loader  = DataLoader(val_ds,  batch_size=batch, shuffle=False, num_workers=nw, pin_memory=True, persistent_workers=(nw>0))
    test_loader = DataLoader(test_ds, batch_size=batch, shuffle=False, num_workers=nw, pin_memory=True, persistent_workers=(nw>0))

    info = {
        "val": len(val_ids),
        "test": len(test_ids),
        "train_tumor": len(train_tumor),
        "train_normal": len(train_normal),
        "labeled_tumor": len(labeled_tumor),
        "unlabeled_tumor": len(unlabeled_tumor),
        "sampling": sampling,
        "normal_weight_per_sample": float(normal_weight),
        "cache": cache_info,
    }
    return train_loader, val_loader, test_loader, info

def train_and_report(raw_dir, proc_dir, pseudo_dir, runs_dir,
                     exp_name, run_mode, p,
                     device="cuda", img_size=512, batch=16, nw=2,
                     lr=1e-3, epochs=20, pos_weight=10.0,
                     thr_monitor=0.5, sampling="auto_group_balance", normal_weight=1.0,
                     pseudo_subdir="sam_box_oracle",
                     thr_select=True, max_fp_rate=0.6, seed=2026, resume=True):

    out_dir = os.path.join(runs_dir, exp_name)
    os.makedirs(out_dir, exist_ok=True)

    train_loader, val_loader, test_loader, info = build_loaders(
        proc_dir=proc_dir, raw_dir=raw_dir, pseudo_dir=pseudo_dir,
        run_mode=run_mode, p=p,
        img_size=img_size, batch=batch, nw=nw,
        sampling=sampling, normal_weight=normal_weight,
        seed=seed, pseudo_subdir=pseudo_subdir
    )

    model = UNet(1,1,32).to(device)
    fit_posw_resume(model, train_loader, val_loader, device, out_dir,
                    pos_weight=pos_weight, lr=lr, epochs_total=epochs, thr_val=thr_monitor, resume=resume)

    best_path = os.path.join(out_dir, "best.pth")
    best = UNet(1,1,32).to(device)
    best.load_state_dict(torch.load(best_path, map_location=device))

    chosen = {"thr": thr_monitor}
    if thr_select:
        thrs = np.linspace(0.1, 0.9, 17)
        rows = sweep_threshold(best, val_loader, device, thrs)
        pick = pick_threshold(rows, max_fp_rate=max_fp_rate)
        if pick is not None:
            chosen = pick

    thr = float(chosen["thr"])

    val_met  = eval_all_and_tumor_only(best, val_loader, device, thr=thr)
    test_met = eval_all_and_tumor_only(best, test_loader, device, thr=thr)
    val_fp   = fp_on_normals(best, val_loader, device, thr=thr)
    test_fp  = fp_on_normals(best, test_loader, device, thr=thr)

    result = {
        "exp": exp_name,
        "mode": run_mode,
        "p": p,
        "pos_weight": pos_weight,
        "sampling": sampling,
        "normal_weight_per_sample": info["normal_weight_per_sample"],
        "thr": thr,
        "max_fp_rate": max_fp_rate,
        **{f"val_{k}": v for k,v in val_met.items()},
        **{f"test_{k}": v for k,v in test_met.items()},
        **{f"val_{k}": v for k,v in val_fp.items()},
        **{f"test_{k}": v for k,v in test_fp.items()},
    }
    return result, info, chosen
