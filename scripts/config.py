from pathlib import Path

# =========================
# Paths
# =========================

ROOT = Path(__file__).resolve().parent

DATA_DIR = ROOT / "data"

TRAIN_CSV = DATA_DIR / "train.csv"
VAL_CSV = DATA_DIR / "val.csv"

CHECKPOINT_DIR = ROOT / "checkpoints"
RESULTS_DIR = ROOT / "results"

CHECKPOINT_DIR.mkdir(exist_ok=True)
RESULTS_DIR.mkdir(exist_ok=True)

# =========================
# Model
# =========================

MODEL_NAME = "facebook/dinov2-base"

EMBED_DIM = 256

IMG_SIZE = 518

# =========================
# Training
# =========================

EPOCHS = 50

BATCH_SIZE = 8

GRAD_ACCUM = 4

NUM_WORKERS = 4

PIN_MEMORY = True

DEVICE = "cuda"

# =========================
# Optimizer
# =========================

BACKBONE_LR = 5e-5

HEAD_LR = 1e-3

WEIGHT_DECAY = 1e-2

# =========================
# ArcFace
# =========================

NUM_SUBCENTERS = 3

MARGIN = 0.5

SCALE = 64.0

# =========================
# Scheduler
# =========================

WARMUP_EPOCHS = 3

MIN_LR = 1e-6

# =========================
# Checkpointing
# =========================

SAVE_EVERY = 1

EARLY_STOPPING = 10

# =========================
# Embeddings
# =========================

NORMALIZE_EMBEDDINGS = True

# =========================
# Random Seed
# =========================

SEED = 42