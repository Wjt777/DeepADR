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

## Run

Classification:

```bash
python train.py --task classification --data_dir data --epochs 25 --batch_size 32
```

Regression:

```bash
python train.py --task regression --data_dir data --epochs 25 --batch_size 32
```

## Reproducibility Notes

- The default classification negative sampling ratio is 1:1.
- The KAN layer runs on the device specified by `--device`; CPU is the default.
