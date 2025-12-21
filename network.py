import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset
from kan import KAN


# =========================
# Dataset
# =========================
class DrugADRDataset(Dataset):
    def __init__(
        self,
        drug_molformer_df,
        drug_target_df,
        adr_biobert_df,
        adr_drug_mtx_df,
    ):
        """
        drug_molformer_df: [n_drugs, mol_dim]
        drug_target_df:    [n_drugs, target_dim]
        adr_biobert_df:    [n_adrs, adr_dim]
        adr_drug_mtx_df:   [n_drugs, n_adrs]
        """

        drug_mol = drug_molformer_df.values
        drug_tar = drug_target_df.values
        adr_feat = adr_biobert_df.T.values
        mtx = adr_drug_mtx_df.values

        drug_idx, adr_idx = np.nonzero(mtx)
        y = mtx[drug_idx, adr_idx]

        self.drug_mol = torch.tensor(drug_mol[drug_idx], dtype=torch.float32)
        self.drug_tar = torch.tensor(drug_tar[drug_idx], dtype=torch.float32)
        self.adr_feat = torch.tensor(adr_feat[adr_idx], dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.float32)

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        return (
            self.drug_mol[idx],
            self.drug_tar[idx],
            self.adr_feat[idx],
            self.y[idx],
        )


# =========================
# Attention
# =========================
class AttentionMechanism(nn.Module):
    def __init__(self, drug_dim, adr_dim, hidden_dim=128):
        super().__init__()
        self.drug_proj = nn.Linear(drug_dim, hidden_dim)
        self.adr_proj = nn.Linear(adr_dim, hidden_dim)
        self.attn = nn.Linear(hidden_dim, 1)

    def forward(self, drug_feat, adr_feat):
        h = torch.tanh(self.drug_proj(drug_feat) + self.adr_proj(adr_feat))
        alpha = torch.sigmoid(self.attn(h))
        attended_drug = drug_feat * alpha
        return torch.cat([attended_drug, adr_feat], dim=1)


# =========================
# CNN Branch
# =========================
class CNNBranch(nn.Module):
    def __init__(self, input_dim, output_dim):
        super().__init__()
        self.conv1 = nn.Conv1d(1, 8, kernel_size=3, padding=1)
        self.conv2 = nn.Conv1d(8, 16, kernel_size=3, padding=1)
        self.pool = nn.MaxPool1d(2)

        fc_dim = (input_dim // 4) * 16
        self.fc1 = nn.Linear(fc_dim, 1024)
        self.fc2 = nn.Linear(1024, output_dim)

    def forward(self, x):
        x = x.unsqueeze(1)
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.flatten(1)
        x = F.relu(self.fc1(x))
        return self.fc2(x)


# =========================
# VAE Branch
# =========================
class VAEBranch(nn.Module):
    def __init__(self, input_dim, latent_dim):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 512),
            nn.ReLU(),
            nn.Linear(512, 64),
            nn.ReLU(),
        )
        self.fc_mu = nn.Linear(64, latent_dim)
        self.fc_logvar = nn.Linear(64, latent_dim)

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def forward(self, x):
        h = self.encoder(x)
        mu = self.fc_mu(h)
        logvar = self.fc_logvar(h)
        return self.reparameterize(mu, logvar)


# =========================
# DeepADR
# =========================
class DeepADR(nn.Module):
    def __init__(
        self,
        mol_dim,
        target_dim,
        adr_dim,
        cnn_out_dim,
        vae_latent_dim,
        kan_hidden_dim,
        kan_device="cpu",
    ):
        super().__init__()

        self.attention = AttentionMechanism(
            drug_dim=mol_dim,
            adr_dim=adr_dim,
        )

        attn_out_dim = mol_dim + adr_dim

        self.cnn_branch = CNNBranch(attn_out_dim, cnn_out_dim)
        self.vae_branch = VAEBranch(target_dim, vae_latent_dim)

        self.kan = KAN(
            width=[cnn_out_dim + vae_latent_dim, kan_hidden_dim, 1],
            grid=2,
            k=3,
            seed=42,
            device=kan_device,
        )

    def forward(self, mol, tar, adr):
        x_attn = self.attention(mol, adr)
        x_cnn = self.cnn_branch(x_attn)
        x_vae = self.vae_branch(tar)
        x = torch.cat([x_cnn, x_vae], dim=1)
        return self.kan(x).squeeze()


# =========================
# RMSE Loss
# =========================
class RMSELoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.mse = nn.MSELoss()

    def forward(self, y_pred, y_true):
        return torch.sqrt(self.mse(y_pred, y_true))


# =========================
# Train
# =========================
def train_model(model, loader, criterion, optimizer, device, epochs):
    model.train()
    for _ in range(epochs):
        for mol, tar, adr, y in loader:
            mol = mol.to(device)
            tar = tar.to(device)
            adr = adr.to(device)
            y = y.to(device)

            optimizer.zero_grad()
            pred = model(mol, tar, adr)
            loss = criterion(pred, y)
            loss.backward()
            optimizer.step()


# =========================
# Evaluate
# =========================
def evaluate_model(model, loader, criterion, device):
    model.eval()
    total_rmse = 0.0

    with torch.no_grad():
        for mol, tar, adr, y in loader:
            mol = mol.to(device)
            tar = tar.to(device)
            adr = adr.to(device)
            y = y.to(device)

            pred = model(mol, tar, adr)
            total_rmse += criterion(pred, y).item()

    avg_rmse = total_rmse / len(loader)

    print("Test Performance:")
    print(f"RMSE: {avg_rmse:.4f}")

    return avg_rmse

