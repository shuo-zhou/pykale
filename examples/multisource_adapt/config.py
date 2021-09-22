"""
Default configurations for multi-source domain adapation
"""
import os

from yacs.config import CfgNode as CN

# import os

# -----------------------------------------------------------------------------
# Config definition
# -----------------------------------------------------------------------------

_C = CN()

# -----------------------------------------------------------------------------
# Dataset
# -----------------------------------------------------------------------------
_C.DATASET = CN()
_C.DATASET.ROOT = "../data"
_C.DATASET.NAME = "digits"  # dset choices=['office', 'image-clef', 'office-home']
# _C.DATASET.SOURCE = ["cartoon", "art_painting", "photo"]
# _C.DATASET.TARGET = ["sketch"]
_C.DATASET.TARGET = "MNIST"
_C.DATASET.NUM_CLASSES = 10
_C.DATASET.NUM_REPEAT = 10  # 10
_C.DATASET.NUM_CHANNELS = 3
_C.DATASET.DIMENSION = 784
_C.DATASET.WEIGHT_TYPE = "natural"
_C.DATASET.SIZE_TYPE = "source"
# ---------------------------------------------------------------------------- #
# Solver
# ---------------------------------------------------------------------------- #
_C.SOLVER = CN()
_C.SOLVER.SEED = 2020
_C.SOLVER.BASE_LR = 0.001  # Initial learning rate
_C.SOLVER.MOMENTUM = 0.9
_C.SOLVER.WEIGHT_DECAY = 0.0005  # 1e-4
_C.SOLVER.NESTEROV = True

_C.SOLVER.TYPE = "SGD"
_C.SOLVER.MAX_EPOCHS = 120  # "nb_adapt_epochs": 100,
# _C.SOLVER.WARMUP = True
_C.SOLVER.MIN_EPOCHS = 20  # "nb_init_epochs": 20,
_C.SOLVER.TRAIN_BATCH_SIZE = 100  # 150
_C.SOLVER.TEST_BATCH_SIZE = 100  # No difference in ADA

# Adaptation-specific solver config
_C.SOLVER.AD_LAMBDA = True
_C.SOLVER.AD_LR = True
_C.SOLVER.INIT_LAMBDA = 1

# ---------------------------------------------------------------------------- #
# Domain Adaptation Net (DAN) configs
# ---------------------------------------------------------------------------- #
_C.DAN = CN()
_C.DAN.METHOD = "M3SDA"  # choices=['CDAN', 'CDAN-E', 'DANN']
_C.DAN.USERANDOM = False
_C.DAN.RANDOM_DIM = 1024
# ---------------------------------------------------------------------------- #
# Misc options
# ---------------------------------------------------------------------------- #
_C.OUTPUT = CN()
_C.OUTPUT.ROOT = "./outputs"  # output_dir
_C.OUTPUT.VERBOSE = False  # To discuss, for HPC jobs
_C.OUTPUT.PB_FRESH = 0  # 0 # 50 # 0 to disable  ; MAYBE make it a command line option
_C.OUTPUT.TB_DIR = os.path.join("lightning_logs", "Tgt" + _C.DATASET.TARGET)
# _C.OUTPUT.DIR = os.path.join(_C.OUTPUT.ROOT, _C.DATASET.NAME + '_rest2' + _C.DATASET.TARGET[0])


def get_cfg_defaults():
    return _C.clone()