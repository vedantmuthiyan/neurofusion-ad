# NeuroFusion-AD

**Multimodal Graph Neural Network for Alzheimer's Disease Progression Prediction**

NeuroFusion-AD is a clinical decision support system that combines plasma biomarkers,
digital speech and gait features, and clinical demographics through a cross-modal
attention fusion architecture to predict Alzheimer's disease progression in MCI patients.

## Clinical Performance

| Dataset | N | AUC | Sensitivity | Specificity |
|---------|---|-----|------------|-------------|
| ADNI (internal test) | 75 | 0.890 | 79.3% | 93.3% |
| Bio-Hermes-001 (external) | 142 | 0.907 | 90.2% | 87.9% |

## Quick Start

```bash
# Run the FHIR API
docker-compose up

# Health check
curl http://localhost:8000/health

# Run clinical demo
docker-compose -f demo/docker-compose.demo.yml up
# Access at http://localhost:3000
```

## Documentation

- [Clinical Validation Report](docs/clinical/CVR_v2.0.md)
- [Technical Due Diligence](docs/investor/technical_due_diligence.md)
- [API Reference](http://localhost:8000/docs)
- [Model Architecture](CLAUDE.md)

## Regulatory

Designed for FDA De Novo and EU MDR Class IIa regulatory submissions.
IEC 62304 compliant software development lifecycle.
ISO 14971 risk management.

## Datasets

- ADNI: Alzheimer's Disease Neuroimaging Initiative
- Bio-Hermes-001: Roche-partnered prospective cohort (N=945)

## License

This software is intended as Clinical Decision Support (CDS) only.
Not a substitute for clinical judgment. For research use.
