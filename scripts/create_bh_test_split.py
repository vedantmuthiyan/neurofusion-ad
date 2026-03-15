"""Create Bio-Hermes-001 stratified 70/15/15 train/val/test split.

Phase 2B checklist item: create biohermes001_test.csv (N≈142).

Phase 2 used only train + val for Bio-Hermes-001. This script creates a held-out
test set using stratified splitting on AMYLOID_POSITIVE to maintain class balance.

Inputs:
    data/processed/biohermes/biohermes001_train.csv
    data/processed/biohermes/biohermes001_val.csv

Outputs:
    data/processed/biohermes/biohermes001_train.csv   (70% of original total)
    data/processed/biohermes/biohermes001_val.csv     (15% of original total)
    data/processed/biohermes/biohermes001_test.csv    (15% of original total, NEW)

Usage:
    python scripts/create_bh_test_split.py

Document traceability:
    Phase 2B Checklist — item 8
    SRS-001 § 5.4 — Data Pipeline
"""
from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split
import structlog

log = structlog.get_logger(__name__)

BH_DIR = Path("data/processed/biohermes")
TRAIN_CSV = BH_DIR / "biohermes001_train.csv"
VAL_CSV = BH_DIR / "biohermes001_val.csv"
TEST_CSV = BH_DIR / "biohermes001_test.csv"

RANDOM_SEED = 42
VAL_FRACTION = 0.15 / 0.85   # 15% of the train+val pool
TEST_FRACTION = 0.15           # 15% of the full dataset


def main() -> None:
    """Rebuild the Bio-Hermes-001 splits with a held-out test set."""
    if not TRAIN_CSV.exists() or not VAL_CSV.exists():
        log.error(
            "create_bh_test_split: input files missing",
            train_csv=str(TRAIN_CSV),
            val_csv=str(VAL_CSV),
        )
        print(
            "ERROR: Bio-Hermes processed CSVs not found. "
            "Run biohermes_preprocessing.py first.\n"
            f"Expected: {TRAIN_CSV} and {VAL_CSV}",
            file=sys.stderr,
        )
        sys.exit(1)

    # Load existing train and val splits and pool them
    df_train = pd.read_csv(TRAIN_CSV)
    df_val = pd.read_csv(VAL_CSV)
    df_all = pd.concat([df_train, df_val], ignore_index=True)

    n_total = len(df_all)
    log.info(
        "create_bh_test_split: loaded pooled dataset",
        n_total=n_total,
        n_train_original=len(df_train),
        n_val_original=len(df_val),
    )

    # Determine stratification column (AMYLOID_POSITIVE if available, else None)
    strat_col = None
    if "AMYLOID_POSITIVE" in df_all.columns:
        n_valid_labels = df_all["AMYLOID_POSITIVE"].notna().sum()
        if n_valid_labels > 10:
            strat_col = df_all["AMYLOID_POSITIVE"].round(0).astype("Int64")
            strat_col = strat_col.where(strat_col.notna(), other=0)
            log.info(
                "create_bh_test_split: stratifying on AMYLOID_POSITIVE",
                n_positive=int((strat_col == 1).sum()),
                n_negative=int((strat_col == 0).sum()),
            )

    # Split: (train+val) vs test
    df_trainval, df_test = train_test_split(
        df_all,
        test_size=TEST_FRACTION,
        random_state=RANDOM_SEED,
        stratify=strat_col,
        shuffle=True,
    )

    # Further split train+val into train and val
    strat_tv = None
    if strat_col is not None:
        strat_tv = strat_col.iloc[df_trainval.index.tolist()].reset_index(drop=True)
        df_trainval = df_trainval.reset_index(drop=True)

    df_train_new, df_val_new = train_test_split(
        df_trainval,
        test_size=VAL_FRACTION,
        random_state=RANDOM_SEED,
        stratify=strat_tv,
        shuffle=True,
    )

    n_train_new = len(df_train_new)
    n_val_new = len(df_val_new)
    n_test = len(df_test)

    log.info(
        "create_bh_test_split: final split sizes",
        n_total=n_total,
        n_train=n_train_new,
        n_val=n_val_new,
        n_test=n_test,
        train_pct=round(n_train_new / n_total * 100, 1),
        val_pct=round(n_val_new / n_total * 100, 1),
        test_pct=round(n_test / n_total * 100, 1),
    )

    # Overwrite train and val; create test
    df_train_new.to_csv(TRAIN_CSV, index=False)
    df_val_new.to_csv(VAL_CSV, index=False)
    df_test.to_csv(TEST_CSV, index=False)

    print(
        f"Bio-Hermes-001 splits written:\n"
        f"  train: {n_train_new:5d} samples → {TRAIN_CSV}\n"
        f"  val:   {n_val_new:5d} samples → {VAL_CSV}\n"
        f"  test:  {n_test:5d} samples → {TEST_CSV}  (NEW — held out)\n"
        f"\nPhase 2B gate: Bio-Hermes test AUC ≥ 0.75"
    )


if __name__ == "__main__":
    main()
