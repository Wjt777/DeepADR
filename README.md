# DeepADR

## Multi-modal Prediction of Adverse Drug Reaction Frequency by Integrating Early-Stage Drug Discovery Information via Kolmogorov–Arnold Networks

This repository provides the model architecture, feature construction pipeline, and inference workflow for ADR frequency prediction from early-stage drug attributes.

## Required Dependencies

| Package | Version |
|--------|----------|
| Python | 3.8.20 |
| PyTorch | 2.4.1 |
| scikit-learn | 1.3.2 |
| pandas | 2.0.3 |
| numpy | 1.24.3 |
| matplotlib | 3.7.2 |
| pykan | 0.2.8 |

## Run the Model

```bash
python run_model.py --data_dir "./data" --epochs 30

