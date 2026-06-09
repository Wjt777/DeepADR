# DeepADR

This directory contains the complete DeepADR implementation for two tasks:

- ADR occurrence classification
- ADR frequency regression

Both tasks use the same multimodal DeepADR backbone: MoLFormer drug features, drug target features, ADR semantic features, attention-based drug-ADR fusion, CNN branch, VAE branch, and KAN output layer.

## Requirements

The code was prepared for Python 3.8.

```bash
pip install -r requirements.txt
```

## Data

Place the following files under `data/`:

```text
data/
  drug_molformer.npy
  drug_target.npy
  adr_semantic.npy
  drug_adr_mtx.csv
```

Expected dimensions for the released data are:

```text
drug_molformer.npy: 719 x 768
drug_target.npy:    719 x 626
adr_semantic.npy:   768 x 994
drug_adr_mtx.csv:   719 x 994
```

For classification, matrix entries equal to `0` are treated as negative samples, and all non-zero frequency levels are treated as positive samples. Negative samples are randomly drawn from zero entries using the `--negative_ratio` argument.

For regression, all non-zero drug-ADR pairs are used with their original frequency labels.

## Run

Classification:

```bash
python train.py --task classification --data_dir data --epochs 25 --batch_size 32
```

Regression:

```bash
python train.py --task regression --data_dir data --epochs 25 --batch_size 32
```

Outputs are saved under `outputs/`:

```text
deepadr_classification_best.pt
deepadr_classification_metrics.json
deepadr_regression_best.pt
deepadr_regression_metrics.json
```

The classification metrics are AUROC, AUPR, and F1. The regression metrics are RMSE, MAE, and PCC.

## Reproducibility Notes

- The default random seed is `42`.
- The default split is 70% training, 20% validation, and 10% testing.
- The default classification negative sampling ratio is 1:1.
- The KAN layer runs on the device specified by `--device`; CPU is the default.
