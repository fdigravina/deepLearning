import os

ROOT = os.path.dirname(os.path.abspath(__file__))
RESULTS_DIR = os.path.join(ROOT, "results")
FIG_DIR = os.path.join(RESULTS_DIR, "figures")

SEED = 0
EPOCHS = 20
LR = 1e-2
WEIGHT_DECAY = 1e-4
BATCH_SIZE = 256
BATCH_SIZE_LARGE = 512

DATASETS = ["spambase", "usps", "letter", "satimage"]
BENCH_DEPTH = 10