import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset
from scipy.stats import pearsonr
from kan import KAN


class AttentionMechanism(nn.Module):
    def __init__(self, drug_dim, adr_dim, hidden_dim=128):
        super().__init__()
        self.drug_projection = nn.Linear(drug_dim, hidden_dim)
        self.adr_projection = nn.Linear(adr_dim, hidden_dim)
        self.attention = nn.Linear(hidden_dim, 1)

    def forward(self, drug_features, adr_features):
        # project into same latent space
        drug_proj = self.drug_projection(drug_features)
        adr_proj = self.adr_projection(adr_features)

        energy = torch.tanh(drug_proj + adr_proj)
        attention_scores = F.softmax(self.attention(energy), dim=1)

        attended_drug = drug_features * attention_scores
        return torch.cat([attended_drug, adr_features], dim=1)


class DrugADRDataset(Dataset):
    def __init__(self, drug_molformer_df, drug_target_df, adr_biobert_df, adr_drug_mtx_df):
        """
        drug_molformer_df: DataFrame, shape [n_drugs, mol_dim]
        drug_target_df:    DataFrame, shape [n_drugs, target_dim]
        adr_biobert_df:    DataFrame, shape [n_adrs, adr_dim]
        adr_drug_mtx_df:   DataFrame, shape [n_drugs, n_adrs]
        """
        self.drug_molformer_mtx = drug_molformer_df.values
        self.drug_target_mtx = drug_target_df.values
        self.side_features = adr_biobert_df.values  

        non_zero_indices = np.nonzero(adr_drug_mtx_df.values)
        self.drug_indices, self.adr_indices = non_zero_indices
        self.y = adr_drug_mtx_df.values[self.drug_indices, self.adr_indices]

        self.drug_mol = torch.tensor(self.drug_molformer_mtx[self.drug_indices], dtype=torch.float32)
        self.drug_tar = torch.tensor(self.drug_target_mtx[self.drug_indices], dtype=torch.float32)
        self.adr_bio = torch.tensor(self.side_features[self.adr_indices], dtype=torch.float32)
        self.y = torch.tensor(self.y, dtype=torch.float32).view(-1)

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        return (
            self.drug_mol[idx],
            self.drug_tar[idx],
            self.adr_bio[idx],
            self.y[idx],
        )


class CNNBranch(nn.Module):
    def __init__(self, input_dim, output_dim):
        super().__init__()
        self.conv1 = nn.Conv1d(1, 8, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv1d(8, 16, kernel_size=3, stride=1, padding=1)
        self.pool = nn.MaxPool1d(kernel_size=2, stride=2)

        # after two pools: length = input_dim / 4, channels = 16
        self.fc_input_dim = (input_dim // 4) * 16
        self.fc1 = nn.Linear(self.fc_input_dim, 1024)
        self.fc2 = nn.Linear(1024, output_dim)
        self.relu = nn.ReLU()

    def forward(self, x):
        batch_size = x.size(0)
        x = x.view(batch_size, 1, -1)
        x = self.pool(self.relu(self.conv1(x)))
        x = self.pool(self.relu(self.conv2(x)))
        x = x.view(batch_size, -1)
        x = self.relu(self.fc1(x))
        x = self.fc2(x)
        return x


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
        z = self.reparameterize(mu, logvar)
        return z


class DeepADR_KAN(nn.Module):
    def __init__(
        self,
        input_dim1,
        input_dim2,
        cnn_output_dim,
        vae_latent_dim,
        kan_hidden_dim,
        output_dim,
        kan_device="cpu",
    ):
        super().__init__()

        self.attention = AttentionMechanism(drug_dim=input_dim1, adr_dim=input_dim1)
        attention_output_dim = input_dim1 * 2

        self.cnn_branch = CNNBranch(input_dim=attention_output_dim, output_dim=cnn_output_dim)
        self.vae_branch = VAEBranch(input_dim=input_dim2, latent_dim=vae_latent_dim)

        self.kan_layer = KAN(
            width=[cnn_output_dim + vae_latent_dim, kan_hidden_dim, output_dim],
            grid=2,
            k=3,
            seed=42,
            device=kan_device,
        )

    def forward(self, mol_features, tar_features, adr_features):
        attention_output = self.attention(mol_features, adr_features)
        cnn_out = self.cnn_branch(attention_output)
        vae_out = self.vae_branch(tar_features)
        combined_features = torch.cat([cnn_out, vae_out], dim=1)
        kan_output = self.kan_layer(combined_features)
        return kan_output


class RMSELoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.mse = nn.MSELoss()

    def forward(self, y_pred, y_true):
        return torch.sqrt(self.mse(y_pred, y_true))


class MAELoss(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, y_pred, y_true):
        return torch.mean(torch.abs(y_pred - y_true))


def train_model(model, train_loader, val_loader, criterion, optimizer, device, num_epochs=10):

    mae_criterion = MAELoss()

    for _ in range(num_epochs):
        model.train()
        for mol, tar, adr, target in train_loader:
            mol = mol.to(device)
            tar = tar.to(device)
            adr = adr.to(device)
            target = target.to(device)

            optimizer.zero_grad()
            output = model(mol, tar, adr)
            loss = criterion(output.squeeze(), target)
            loss.backward()
            optimizer.step()


def evaluate_model(model, test_loader, criterion, device):
    mae_criterion = MAELoss()

    model.eval()
    test_rmse = 0.0
    test_mae = 0.0
    preds = []
    targets = []

    with torch.no_grad():
        for mol, tar, adr, target in test_loader:
            mol = mol.to(device)
            tar = tar.to(device)
            adr = adr.to(device)
            target = target.to(device)

            output = model(mol, tar, adr)

            test_rmse += criterion(output.squeeze(), target).item()
            test_mae += mae_criterion(output.squeeze(), target).item()
            preds.extend(output.squeeze().cpu().numpy())
            targets.extend(target.cpu().numpy())

    avg_rmse = test_rmse / len(test_loader)
    avg_mae = test_mae / len(test_loader)
    pcc, _ = pearsonr(targets, preds)

    print("Test Performance:")
    print(f"RMSE: {avg_rmse:.4f}")
    print(f"MAE:  {avg_mae:.4f}")
    print(f"PCC:  {pcc:.4f}")
