import os, glob
import cv2
from tqdm import tqdm

def read_list(p: str) -> list[str]:
    return [x.strip() for x in open(p) if x.strip()]

def write_list(p: str, lst: list[str]) -> None:
    with open(p, "w") as f:
        for x in lst:
            f.write(x + "\n")

def stem(x: str) -> str:
    return os.path.splitext(x)[0]

def is_tumor_from_gt_mask(gt_masks_dir: str, img_id: str) -> bool:
    mpath = os.path.join(gt_masks_dir, stem(img_id) + ".png")
    m = cv2.imread(mpath, 0)
    if m is None:
        raise FileNotFoundError(mpath)
    return (m > 0).any()

def build_train_pool(raw_images_dir: str, val_ids: list[str], test_ids: list[str]) -> list[str]:
    img_files = glob.glob(os.path.join(raw_images_dir, "*"))
    all_ids = [os.path.basename(p) for p in img_files]
    val_set, test_set = set(val_ids), set(test_ids)
    return [i for i in all_ids if i not in val_set and i not in test_set]

def load_or_build_tumor_normal_lists(proc_dir: str, raw_images_dir: str, gt_masks_dir: str,
                                     val_ids: list[str], test_ids: list[str],
                                     cache_dir: str | None = None,
                                     show_progress: bool = True):
    cache_dir = cache_dir or os.path.join(proc_dir, "splits")
    os.makedirs(cache_dir, exist_ok=True)
    cache_tumor = os.path.join(cache_dir, "train_tumor_all.txt")
    cache_normal = os.path.join(cache_dir, "train_normal_all.txt")

    if os.path.exists(cache_tumor) and os.path.exists(cache_normal):
        train_tumor = read_list(cache_tumor)
        train_normal = read_list(cache_normal)
        return train_tumor, train_normal, {"cached": True, "cache_tumor": cache_tumor, "cache_normal": cache_normal}

    train_pool = build_train_pool(raw_images_dir, val_ids, test_ids)
    it = tqdm(train_pool, desc="Scan masks") if show_progress else train_pool

    train_tumor, train_normal = [], []
    for img_id in it:
        if is_tumor_from_gt_mask(gt_masks_dir, img_id):
            train_tumor.append(img_id)
        else:
            train_normal.append(img_id)

    write_list(cache_tumor, train_tumor)
    write_list(cache_normal, train_normal)

    return train_tumor, train_normal, {"cached": False, "cache_tumor": cache_tumor, "cache_normal": cache_normal}

def budget_split(train_tumor_ids: list[str], p: float, seed: int = 2026):
    import numpy as np
    rng = np.random.default_rng(seed + int(p * 1000))
    n = len(train_tumor_ids)
    k = max(1, int(round(n * p)))
    idx = rng.choice(n, size=k, replace=False)
    labeled = [train_tumor_ids[i] for i in idx]
    labeled_set = set(labeled)
    unlabeled = [x for x in train_tumor_ids if x not in labeled_set]
    return labeled, unlabeled
