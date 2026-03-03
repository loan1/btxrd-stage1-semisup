"""
Project config for BTXRD Stage-1 (Semi-supervised + SAM/MedSAM + U-Net)
"""
DRIVE_ROOT = "/content/drive/MyDrive/BTXRD"

RAW_DIR    = f"{DRIVE_ROOT}/raw"
PROC_DIR   = f"{DRIVE_ROOT}/processed"
PSEUDO_DIR = f"{DRIVE_ROOT}/pseudo"
RUNS_DIR   = f"{DRIVE_ROOT}/runs"

DEFAULT_PSEUDO_SUBDIR = "sam_box_oracle"

DEFAULT_IMG_SIZE = 512
DEFAULT_BATCH = 16
DEFAULT_NUM_WORKERS = 2
DEFAULT_LR = 1e-3
DEFAULT_EPOCHS = 20
DEFAULT_POS_WEIGHT = 10.0
DEFAULT_SEED = 2026
