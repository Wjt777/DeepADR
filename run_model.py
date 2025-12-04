import argparse
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, random_split
from network import DrugADRDataset, DeepADR_MLP, RMSELoss, train_model, evaluate_model

def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

set_seed(42)

def parse_args():
    parser = argparse.ArgumentParser(description="Run DeepADR model")
    parser.add_argument("--data_dir", type=str, required=True)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-3)
    return parser.parse_args()


def main():
    args = parse_args()

    # Device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # ===============================
    # Load data (now using .npy)
    # ===============================
    drug_mol = np.load(f"{args.data_dir}/drug_molformer.npy")
    drug_tar = np.load(f"{args.data_dir}/drug_target.npy")
    adr_bio = np.load(f"{args.data_dir}/adr_semantic.npy")
    adr_mtx = pd.read_csv(f"{args.data_dir}/drug_adr_mtx.csv", index_col=0)

    # Convert numpy back to DataFrame for dataset format
    drug_mol_df = pd.DataFrame(drug_mol)
    drug_tar_df = pd.DataFrame(drug_tar)
    adr_bio_df = pd.DataFrame(adr_bio)

    # ===============================
    # Dataset & split
    # ===============================
    dataset = DrugADRDataset(drug_mol_df, drug_tar_df, adr_bio_df, adr_mtx)

    train_len = int(0.7 * len(dataset))
    val_len = int(0.2 * len(dataset))
    test_len = len(dataset) - train_len - val_len

    train_set, val_set, test_set = random_split(dataset, [train_len, val_len, test_len])

    train_loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_set, batch_size=args.batch_size, shuffle=False)
    test_loader = DataLoader(test_set, batch_size=args.batch_size, shuffle=False)

    # ===============================
    # Build model
    # ===============================
    model = DeepADR_MLP(
        input_dim1=drug_mol_df.shape[1],
        input_dim2=drug_tar_df.shape[1],
        cnn_output_dim=128,
        vae_latent_dim=128,
        mlp_hidden_dim=32,
        output_dim=1,
    ).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    criterion = RMSELoss()

    # ===============================
    # Train
    # ===============================
    train_model(
        model, train_loader, val_loader, criterion, optimizer, device, epochs=args.epochs
    )

    # ===============================
    # Final Evaluation
    # ===============================
    print("\n======= Final Performance =======")
    evaluate_model(model, test_loader, criterion, device)

    # ===============================
    # Save model
    # ===============================
    torch.save(model.state_dict(), f"{args.data_dir}/deepadr_model.pth")
    print(f"Model saved → {args.data_dir}/deepadr_model.pth")


if __name__ == "__main__":
    main()
