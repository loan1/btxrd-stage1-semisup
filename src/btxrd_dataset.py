import os, glob
import cv2
import numpy as np
from torch.utils.data import Dataset

def _stem(x: str) -> str:
    return os.path.splitext(x)[0]

def find_image_path(images_dir: str, image_id: str) -> tuple[str, str]:
    stem = _stem(image_id)
    cand = glob.glob(os.path.join(images_dir, stem + ".*"))
    if not cand:
        raise FileNotFoundError(f"Missing image for {image_id} (stem={stem}) in {images_dir}")
    cand_sorted = sorted(cand, key=lambda p: (os.path.splitext(p)[1].lower() not in [".jpg",".jpeg"], p))
    return cand_sorted[0], stem

class BTXRDSegDataset(Dataset):
    """
    mask_mode:
      - 'gt'    : ground truth masks from gt_masks_dir/<stem>.png
      - 'pseudo': pseudo masks from pseudo_masks_dir/<stem>.png (if missing -> empty mask)
    """
    def __init__(self,
                 images_dir: str,
                 ids_list: list[str],
                 mask_mode: str,
                 gt_masks_dir: str,
                 pseudo_masks_dir: str | None = None,
                 img_size: int = 512):
        assert mask_mode in ("gt", "pseudo")
        if mask_mode == "pseudo":
            assert pseudo_masks_dir is not None, "pseudo_masks_dir is required for mask_mode='pseudo'"
        self.images_dir = images_dir
        self.ids_list = ids_list
        self.mask_mode = mask_mode
        self.gt_masks_dir = gt_masks_dir
        self.pseudo_masks_dir = pseudo_masks_dir
        self.img_size = int(img_size)

    def __len__(self) -> int:
        return len(self.ids_list)

    def _load_mask(self, stem: str, img_shape: tuple[int,int]) -> np.ndarray:
        if self.mask_mode == "gt":
            mpath = os.path.join(self.gt_masks_dir, stem + ".png")
            m = cv2.imread(mpath, 0)
            if m is None:
                raise FileNotFoundError(mpath)
            return m

        ppath = os.path.join(self.pseudo_masks_dir, stem + ".png")
        m = cv2.imread(ppath, 0)
        if m is None:
            return np.zeros(img_shape, dtype=np.uint8)
        return m

    def __getitem__(self, idx: int):
        image_id = self.ids_list[idx]
        img_path, stem = find_image_path(self.images_dir, image_id)

        img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
        if img is None:
            raise FileNotFoundError(img_path)

        mask = self._load_mask(stem, img.shape)

        img_rs  = cv2.resize(img,  (self.img_size, self.img_size), interpolation=cv2.INTER_LINEAR)
        mask_rs = cv2.resize(mask, (self.img_size, self.img_size), interpolation=cv2.INTER_NEAREST)

        img_rs  = img_rs.astype(np.float32) / 255.0
        mask_rs = (mask_rs > 127).astype(np.float32)

        return img_rs[None, :, :], mask_rs[None, :, :], image_id
