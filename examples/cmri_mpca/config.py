"""
Default configurations for cardiac MRI data (ShefPAH) processing and classification
"""

from yacs.config import CfgNode as CN

# -----------------------------------------------------------------------------
# Config definition
# -----------------------------------------------------------------------------

_C = CN()

# -----------------------------------------------------------------------------
# Dataset
# -----------------------------------------------------------------------------
_C.DATASET = CN()
_C.DATASET.SOURCE = "https://github.com/pykale/data/raw/main/images/ShefPAH-179/SA_64x64_v2.0.zip"
_C.DATASET.ROOT = "../data"
_C.DATASET.IMG_DIR = "DICOM"
_C.DATASET.BASE_DIR = "SA_64x64_v2.0"
_C.DATASET.FILE_FORAMT = "zip"
_C.DATASET.LANDMARK_FILE = "landmarks.csv"
_C.DATASET.MASK_DIR = "Mask"
# ---------------------------------------------------------------------------- #
# Image processing
# ---------------------------------------------------------------------------- #
_C.PROC = CN()
_C.PROC.SCALE = 2

# ---------------------------------------------------------------------------- #
# Visualization
# ---------------------------------------------------------------------------- #
_C.IM_KWARGS = CN()
_C.IM_KWARGS.cmap = "gray"

_C.MARKER_KWARGS = CN()
_C.MARKER_KWARGS.marker = "+"
_C.MARKER_KWARGS.color = "r"
_C.MARKER_KWARGS.s = 100
_C.MARKER_KWARGS.linewidths = 1.5
_C.MARKER_KWARGS.edgecolors = "face"
# see https://matplotlib.org/stable/api/_as_gen/matplotlib.pyplot.scatter.html for more options

_C.WEIGHT_KWARGS = CN()
_C.WEIGHT_KWARGS.markersize = 6
_C.WEIGHT_KWARGS.alpha = 0.7

_C.PLT_KWARGS = CN()
_C.PLT_KWARGS.n_cols = 10

# ---------------------------------------------------------------------------- #
# Machine learning pipeline
# ---------------------------------------------------------------------------- #
_C.PIPELINE = CN()
_C.PIPELINE.CLASSIFIER = "linear_svc"  # ["svc", "linear_svc", "lr"]

# ---------------------------------------------------------------------------- #
# Misc options
# ---------------------------------------------------------------------------- #
_C.OUTPUT = CN()
_C.OUTPUT.ROOT = "./outputs"  # output_dir
_C.OUTPUT.SAVE_FIG = True

_C.SAVE_FIG_KWARGS = CN()
_C.SAVE_FIG_KWARGS.format = "pdf"
_C.SAVE_FIG_KWARGS.bbox_inches = "tight"


def get_cfg_defaults():
    return _C.clone()
