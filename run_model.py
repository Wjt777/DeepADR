import argparse
import os
import random

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, random_split

from network import DrugADRDataset, DeepADR_KAN, RMSELoss, train_model, evaluate_model



def parse_args():
    parser = argparse.ArgumentParser(description="Run DeepADR-KAN model")

    parser.add_argument("--data_dir", type=str, required=True, help="Directory of data files")
    parser.add_argument("--epochs", type=int, default=25)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight_decay", type=float, default=1e-4)

    return parser.parse_args()


def main():
    args = parse_args()
    set_seed(42)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # ---------- load data ----------
    drug_mol = np.load(os.path.join(args.data_dir, "drug_molformer.npy"))
    drug_tar = np.load(os.path.join(args.data_dir, "drug_target.npy"))
    adr_bio = np.load(os.path.join(args.data_dir, "adr_semantic.npy"))
    adr_mtx = pd.read_csv(os.path.join(args.data_dir, "drug_adr_mtx.csv"), index_col=0)

    drug_mol_df = pd.DataFrame(drug_mol)
    drug_tar_df = pd.DataFrame(drug_tar)
    adr_bio_df = pd.DataFrame(adr_bio)

    dataset = DrugADRDataset(drug_mol_df, drug_tar_df, adr_bio_df, adr_mtx)

    # ---------- split ----------
    n_total = len(dataset)
    n_train = int(0.7 * n_total)
    n_val = int(0.2 * n_total)
    n_test = n_total - n_train - n_val

    train_set, val_set, test_set = random_split(dataset, [n_train, n_val, n_test])

    train_loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_set, batch_size=args.batch_size, shuffle=False)
    test_loader = DataLoader(test_set, batch_size=args.batch_size, shuffle=False)

    # ---------- build model ----------
    mol_dim = drug_mol_df.shape[1]
    tar_dim = drug_tar_df.shape[1]

    model = DeepADR_KAN(
        input_dim1=mol_dim,
        input_dim2=tar_dim,
        cnn_output_dim=128,
        vae_latent_dim=128,
        kan_hidden_dim=16,
        output_dim=1,
        kan_device=device,
    ).to(device)

    criterion = RMSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    # ---------- train ----------
    train_model(model, train_loader, val_loader, criterion, optimizer, device, num_epochs=args.epochs)

    # ---------- evaluate (only final performance printed) ----------
    print("\n========== Final Test Performance ==========")
    evaluate_model(model, test_loader, criterion, device)

    # ---------- save model ----------
    save_path = os.path.join(args.data_dir, "deepadr_kan_model.pth")
    torch.save(model.state_dict(), save_path)
    print(f"Model saved to: {save_path}")


if __name__ == "__main__":
    main()
