#!/usr/bin/env python3
"""End-to-end sanity check for NeuroFusion-AD model.

Runs a forward pass with batch_size=16 of synthetic data,
verifies all output shapes, checks for NaN values,
and prints a summary.

Usage:
    python scripts/sanity_check_e2e.py
"""
import sys
sys.path.insert(0, '.')

import torch
from src.models.neurofusion_model import NeuroFusionAD, CLINICAL_DISCLAIMER
from src.data.dataset import generate_synthetic_adni, create_dataloaders


def main() -> None:
    """Run end-to-end sanity check: instantiate model, load synthetic data, forward pass.

    Verifies:
        - Model instantiates without error
        - Synthetic ADNI dataset generates and loads without error
        - Forward pass on a batch of 16 patients completes without error
        - All output tensor shapes are correct
        - No NaN values appear in any output tensor
        - Mandatory clinical disclaimer is present in outputs

    Raises:
        AssertionError: If any shape, NaN, or disclaimer check fails.
        SystemExit: With code 1 if an unexpected exception occurs.
    """
    print("=== NeuroFusion-AD End-to-End Sanity Check ===\n")

    # 1. Instantiate model
    model = NeuroFusionAD()
    model.eval()
    n_params = model.count_parameters()
    print(f"Model instantiated. Parameters: {n_params:,}")

    # 2. Generate synthetic dataset
    dataset = generate_synthetic_adni(n_samples=64, seed=42)
    train_loader, val_loader, test_loader = create_dataloaders(dataset, batch_size=16)
    print(f"Dataset: {len(dataset)} samples. Train batches: {len(train_loader)}")

    # 3. Forward pass
    batch = next(iter(train_loader))
    with torch.no_grad():
        outputs = model(batch)

    # 4. Shape assertions
    B = batch["fluid"].shape[0]
    assert outputs["amyloid_logit"].shape == (B, 1), (
        f"Expected amyloid_logit shape ({B}, 1), got {outputs['amyloid_logit'].shape}"
    )
    assert outputs["mmse_slope"].shape == (B, 1), (
        f"Expected mmse_slope shape ({B}, 1), got {outputs['mmse_slope'].shape}"
    )
    assert outputs["cox_log_hazard"].shape == (B, 1), (
        f"Expected cox_log_hazard shape ({B}, 1), got {outputs['cox_log_hazard'].shape}"
    )
    assert outputs["fused_embedding"].shape == (B, 768), (
        f"Expected fused_embedding shape ({B}, 768), got {outputs['fused_embedding'].shape}"
    )

    # 5. NaN checks
    for key, val in outputs.items():
        if isinstance(val, torch.Tensor):
            assert not torch.isnan(val).any(), f"NaN detected in output '{key}'"

    # 6. Disclaimer check
    assert outputs["disclaimer"] == CLINICAL_DISCLAIMER, (
        f"Disclaimer mismatch. Got: '{outputs['disclaimer']}'"
    )

    print("\nShape assertions: PASSED")
    print("NaN checks: PASSED")
    print("Disclaimer present: PASSED")
    print(f"\nOutput shapes:")
    for key, val in outputs.items():
        if isinstance(val, torch.Tensor):
            print(f"  {key}: {tuple(val.shape)}")
    print(f"\nDisclaimer: \"{outputs['disclaimer']}\"")
    print(f"\n=== SANITY CHECK PASSED ===")


if __name__ == "__main__":
    main()
