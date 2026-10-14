"""Paths and global experiment settings shared by every script."""
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
DATA_RAW = ROOT / "data" / "raw"
DATA_PROCESSED = ROOT / "data" / "processed"
RESULTS = ROOT / "results"
TABLES = RESULTS / "tables"
FIGURES = RESULTS / "figures"
CACHE = RESULTS / "cache"

SEED = 42

# Outer evaluation protocol: repeated stratified k-fold.
CV_FOLDS = 5
CV_REPEATS = 2
# Inner hyper-parameter search.
INNER_FOLDS = 3
N_ITER_SEARCH = 10

for _p in (DATA_PROCESSED, TABLES, FIGURES, CACHE):
    _p.mkdir(parents=True, exist_ok=True)
