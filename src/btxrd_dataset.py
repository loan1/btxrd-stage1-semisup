import os, glob
import cv2
import numpy as np
from torch.utils.data import Dataset

def find_image_path(images_dir, image_id):
    stem = os.path.splitext(image_id)[0]
    cand = glob.glob(os.path.join(images_dir, stem + ".*"))
    if not cand:
        raise FileNotFoundError(f"Missing image for {image_id}")
    return cand[0], stem

class BTXRDSegDataset(Dataset):
    """
    mask_mode:
      - 'gt'    : load ground-truth mask from masks_gt (png)
      - 'pseudo': load pseudo mask if exists, else empty mask
    """
    def __init__(self, images_dir, ids_list, mask_mode,
                 gt_masks_dir, pseudo_masks_dir=None, img_size=512):
        self.images_dir = images_dir
        self.ids_list = ids_list
        self.mask_mode = mask_mode
        self.gt_masks_dir = gt_masks_dir
        self.pseudo_masks_dir = pseudo_masks_dir
        self.img_size = img_size

    def __len__(self):
        return len(self.ids_list)

    def _load_mask(self, stem, img_shape):
        if self.mask_mode == "gt":
            mpath = os.path.join(self.gt_masks_dir, stem + ".png")
            m = cv2.imread(mpath, 0)
            if m is None:
                raise FileNotFoundError(mpath)
            return m

        # pseudo mode
        ppath = os.path.join(self.pseudo_masks_dir, stem + ".png")
        m = cv2.imread(ppath, 0)
        if m is None:
            return np.zeros(img_shape, dtype=np.uint8)
        return m

    def __getitem__(self, idx):
        image_id = self.ids_list[idx]
        img_path, stem = find_image_path(self.images_dir, image_id)

        img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
        if img is None:
            raise FileNotFoundError(img_path)

        mask = self._load_mask(stem, img.shape)

        img_rs  = cv2.resize(img, (self.img_size, self.img_size), interpolation=cv2.INTER_LINEAR)
        mask_rs = cv2.resize(mask, (self.img_size, self.img_size), interpolation=cv2.INTER_NEAREST)

        img_rs  = img_rs.astype(np.float32) / 255.0
        mask_rs = (mask_rs > 127).astype(np.float32)

        return img_rs[None, :, :], mask_rs[None, :, :], image_id
