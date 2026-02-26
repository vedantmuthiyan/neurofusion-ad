# data-engineer-agent Handoff — 2026-02-26

## Completed This Session

- `src/data/validators.py` — InputValidator with full clinical range checks for fluid
  biomarkers (ptau217, abeta42_40_ratio, nfl), acoustic features (jitter, shimmer),
  and clinical features (mmse). Raises ValueError with descriptive messages on failure.
  Logs validation events with structlog — never logs actual feature values (PHI compliance).

- `src/data/adni_preprocessing.py` — ADNIPreprocessor with z-score normalization
  (clipped to [-5, 5]), NaN imputation (mean/median/zero strategies), preprocess_record
  for single records, and preprocess_batch for batched processing. Population-level
  normalization constants are estimated from published literature.

- `src/data/digital_biomarker_synthesis.py` — DigitalBiomarkerSynthesizer with
  synthesize_acoustic (n, 12), synthesize_motor (n, 8), synthesize_fluid_biomarkers
  (n, 6) with two sub-populations (amyloid+/amyloid-), synthesize_clinical (n, 10),
  synthesize_labels (3 output heads), and synthesize_full_dataset convenience wrapper.
  All values are clipped to validated clinical ranges from the specification.

- `src/data/dataset.py` — NeuroFusionDataset (PyTorch Dataset), generate_synthetic_adni
  factory function, and create_dataloaders with reproducible train/val/test splits.
  Patient IDs are SHA-256 hashed immediately on ingestion — raw IDs are never stored.

- `src/data/__init__.py` — Exports all public symbols from the data package.

- `data/__init__.py`, `data/raw/.gitkeep`, `data/processed/.gitkeep` — Placeholders.

- `tests/unit/test_data.py` — 23 unit tests covering all public functions. All pass.

## Decisions Made (with rationale)

- SHA-256 hash patient IDs before any logging: IEC 62304 §6.1 PHI handling compliance.
  Raw patient identifiers must never appear in logs, audit trails, or error messages.

- Clip normalized values to [-5, 5]: Prevents extreme outliers from destabilizing
  gradient flow during model training. Industry standard for biomedical ML pipelines.

- Use `torch.tensor(array.tolist(), dtype=torch.float32)` instead of `torch.from_numpy()`:
  The project environment has NumPy 2.x installed but PyTorch was compiled against
  NumPy 1.x. The `torch.from_numpy()` bridge raises `RuntimeError: Numpy is not available`
  at runtime. Converting to Python list first bypasses the C-level bridge.

- Separate DigitalBiomarkerSynthesizer from dataset.py: Allows independent use of
  the synthesizer for data augmentation, cross-validation fold generation, and
  stress testing without instantiating a full dataset.

- Two-population fluid biomarker model (amyloid+/amyloid-): Reflects real-world
  ADNI cohort structure where ~40% of MCI patients are amyloid-positive. This
  prevents the model from seeing perfectly overlapping class distributions during testing.

- All synthesis methods clip to the validated clinical ranges from CLAUDE.md:
  This ensures that synthetic data always passes InputValidator checks, making
  the test suite self-consistent.

## Current State

- Working: All four data pipeline modules with 23/23 passing unit tests.
- Blocked: Nothing — data pipeline is independent of model architecture.

## Next Session Must Start With

1. Verify tests still pass: `pytest tests/unit/test_data.py -v`
2. If integrating with model: import `generate_synthetic_adni` and `create_dataloaders`
   from `src.data` to generate DataLoaders for end-to-end training loop testing.
3. When real ADNI data arrives (after DUA): update FLUID_MEAN/FLUID_STD and other
   normalization constants in ADNIPreprocessor with population statistics computed
   from the real cohort.

## Open Questions for Human Review

- Real ADNI data access: requires approved DUA at adni.loni.usc.edu. All current
  tests run on synthetic data only.
- Bio-Hermes access: requires registration at globalalzplatform.org AD Workbench.
- NumPy 2.x / PyTorch 2.1.2 incompatibility: The environment has NumPy 2.4.1 but
  torch 2.1.2 was compiled against NumPy 1.x. The workaround (`.tolist()` conversion)
  is in place, but upgrading PyTorch to a version compiled against NumPy 2.x would
  be cleaner for production. Suggest pinning `numpy<2.0` in requirements.txt or
  upgrading torch to >=2.3.0 which supports NumPy 2.x.
