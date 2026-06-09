import argparse
import json
import random
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from scipy.stats import pearsonr
from sklearn.metrics import (
    average_precision_score,
    f1_score,
    mean_absolute_error,
    mean_squared_error,
    roc_auc_score,
)
from torch.utils.data import DataLoader

from deepadr import (
    DeepADR,
    load_data,
    make_classification_dataset,
    make_regression_dataset,
    split_dataset,
)


def parse_args():
    parser = argparse.ArgumentParser(description="Train and evaluate DeepADR.")
    parser.add_argument("--task", choices=["classification", "regression"], required=True)
    parser.add_argument("--data_dir", type=str, default="data")
    parser.add_argument("--output_dir", type=str, default="outputs")
    parser.add_argument("--epochs", type=int, default=25)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight_decay", type=float, default=1e-4)
    parser.add_argument("--negative_ratio", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--kan_hidden_dim", type=int, default=16)
    return parser.parse_args()


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


class RMSELoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.mse = nn.MSELoss()

    def forward(self, prediction, target):
        return torch.sqrt(self.mse(prediction, target))


def build_model(data, args):
    return DeepADR(
        mol_dim=data.drug_molformer.shape[1],
        target_dim=data.drug_target.shape[1],
        adr_dim=data.adr_features.shape[1],
        cnn_output_dim=128,
        vae_latent_dim=128,
        kan_hidden_dim=args.kan_hidden_dim,
        output_dim=1,
        kan_device=args.device,
    ).to(args.device)


def train_one_epoch(model, loader, criterion, optimizer, device, task):
    model.train()
    running_loss = 0.0
    for mol, target, adr, y in loader:
        mol = mol.to(device)
        target = target.to(device)
        adr = adr.to(device)
        y = y.to(device)

        optimizer.zero_grad()
        output = model(mol, target, adr)
        if task == "classification":
            loss = criterion(output, y)
        else:
            loss = criterion(output, y)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        running_loss += loss.item()
    return running_loss / max(len(loader), 1)


def collect_predictions(model, loader, device):
    model.eval()
    predictions = []
    labels = []
    with torch.no_grad():
        for mol, target, adr, y in loader:
            mol = mol.to(device)
            target = target.to(device)
            adr = adr.to(device)
            output = model(mol, target, adr)
            predictions.extend(output.detach().cpu().numpy())
            labels.extend(y.numpy())
    return np.asarray(labels), np.asarray(predictions)


def evaluate_classification(labels, logits):
    probabilities = 1.0 / (1.0 + np.exp(-logits))
    predicted_labels = (probabilities >= 0.5).astype(int)
    return {
        "auroc": float(roc_auc_score(labels, probabilities)),
        "aupr": float(average_precision_score(labels, probabilities)),
        "f1": float(f1_score(labels, predicted_labels)),
    }


def evaluate_regression(labels, predictions):
    rmse = mean_squared_error(labels, predictions, squared=False)
    mae = mean_absolute_error(labels, predictions)
    pcc = pearsonr(labels, predictions)[0]
    return {
        "rmse": float(rmse),
        "mae": float(mae),
        "pcc": float(pcc),
    }


def main():
    args = parse_args()
    set_seed(args.seed)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    data = load_data(args.data_dir)
    if args.task == "classification":
        dataset = make_classification_dataset(
            data,
            negative_ratio=args.negative_ratio,
            seed=args.seed,
        )
        criterion = nn.BCEWithLogitsLoss()
    else:
        dataset = make_regression_dataset(data)
        criterion = RMSELoss()

    train_set, val_set, test_set = split_dataset(dataset, seed=args.seed)
    train_loader = DataLoader(
        train_set,
        batch_size=args.batch_size,
        shuffle=True,
        drop_last=True,
    )
    val_loader = DataLoader(val_set, batch_size=args.batch_size, shuffle=False)
    test_loader = DataLoader(test_set, batch_size=args.batch_size, shuffle=False)

    model = build_model(data, args)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.lr,
        weight_decay=args.weight_decay,
    )

    best_val_loss = float("inf")
    best_model_path = output_dir / f"deepadr_{args.task}_best.pt"

    for epoch in range(1, args.epochs + 1):
        train_loss = train_one_epoch(
            model,
            train_loader,
            criterion,
            optimizer,
            args.device,
            args.task,
        )
        val_labels, val_predictions = collect_predictions(model, val_loader, args.device)
        val_loss = criterion(
            torch.tensor(val_predictions, dtype=torch.float32),
            torch.tensor(val_labels, dtype=torch.float32),
        ).item()

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), best_model_path)

        print(
            f"Epoch {epoch:03d}/{args.epochs} "
            f"train_loss={train_loss:.4f} val_loss={val_loss:.4f}"
        )

    model.load_state_dict(torch.load(best_model_path, map_location=args.device))
    test_labels, test_predictions = collect_predictions(model, test_loader, args.device)

    if args.task == "classification":
        metrics = evaluate_classification(test_labels, test_predictions)
    else:
        metrics = evaluate_regression(test_labels, test_predictions)

    metrics_path = output_dir / f"deepadr_{args.task}_metrics.json"
    with metrics_path.open("w", encoding="utf-8") as handle:
        json.dump(metrics, handle, indent=2)

    print("Test metrics:")
    print(json.dumps(metrics, indent=2))
    print(f"Saved best model to: {best_model_path}")
    print(f"Saved metrics to: {metrics_path}")


if __name__ == "__main__":
    main()
