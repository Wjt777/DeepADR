import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset
import numpy as np
from scipy.stats import pearsonr


class AttentionMechanism(nn.Module):
    def __init__(self, drug_dim, adr_dim, hidden_dim=128):
        super(AttentionMechanism, self).__init__()
        self.drug_projection = nn.Linear(drug_dim, hidden_dim)
        self.adr_projection = nn.Linear(adr_dim, hidden_dim)
        self.attention = nn.Linear(hidden_dim, 1)

    def forward(self, drug_features, adr_features):
        # project to the same latent space
        drug_proj = self.drug_projection(drug_features)
        adr_proj = self.adr_projection(adr_features)

        # attention scores
        energy = torch.tanh(drug_proj + adr_proj)
        attention_scores = F.softmax(self.attention(energy), dim=1)

        # weighted drug features
        attended_drug = drug_features * attention_scores

        # concatenate attended drug and ADR features
        output = torch.cat([attended_drug, adr_features], dim=1)
        return output


class DrugADRDataset(Dataset):
    def __init__(self, drug_molformer, drug_target, adr_biobert, adr_drug_mtx):
        """
        drug_molformer: pd.DataFrame (n_drugs x mol_dim)
        drug_target: pd.DataFrame (n_drugs x target_dim)
        adr_biobert: pd.DataFrame (n_adrs x adr_dim)
        adr_drug_mtx: pd.DataFrame (n_drugs x n_adrs), target values
        """
        # convert dataframes to numpy
        self.drug_molformer_mtx = drug_molformer.values
        self.drug_target_mtx = drug_target.values
        self.side_features = adr_biobert.values.T  # shape: (n_adrs, adr_dim)

        # get non-zero entries (drug, adr) and target y
        non_zero_indices = np.nonzero(adr_drug_mtx.values)
        self.drug_indices, self.adr_indices = non_zero_indices
        self.y = adr_drug_mtx.values[self.drug_indices, self.adr_indices]

        # slice features
        self.drug_mol = self.drug_molformer_mtx[self.drug_indices]
        self.drug_tar = self.drug_target_mtx[self.drug_indices]
        self.adr_bio = self.side_features[self.adr_indices]

        # to tensors
        self.drug_mol = torch.tensor(self.drug_mol, dtype=torch.float32)
        self.drug_tar = torch.tensor(self.drug_tar, dtype=torch.float32)
        self.adr_bio = torch.tensor(self.adr_bio, dtype=torch.float32)
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
        super(CNNBranch, self).__init__()
        self.conv1 = nn.Conv1d(1, 8, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv1d(8, 16, kernel_size=3, stride=1, padding=1)
        self.pool = nn.MaxPool1d(kernel_size=2, stride=2)

        # after two poolings, length is input_dim / 4
        self.fc_input_dim = (input_dim // 4) * 16

        self.fc1 = nn.Linear(self.fc_input_dim, 1024)
        self.fc2 = nn.Linear(1024, output_dim)
        self.relu = nn.ReLU()

        print(f"CNN Branch initialized with fc_input_dim: {self.fc_input_dim}")

    def forward(self, x):
        batch_size = x.size(0)
        x = x.view(batch_size, 1, -1)  # [B, C, L]

        x = self.pool(self.relu(self.conv1(x)))
        x = self.pool(self.relu(self.conv2(x)))

        x = x.view(batch_size, -1)
        x = self.relu(self.fc1(x))
        x = self.fc2(x)
        return x


class VAEBranch(nn.Module):
    def __init__(self, input_dim, latent_dim):
        super(VAEBranch, self).__init__()
        # encoder
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(512, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(256, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(0.2),
        )

        self.fc_mu = nn.Linear(128, latent_dim)
        self.fc_logvar = nn.Linear(128, latent_dim)

        # decoder
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(128, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(256, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(512, input_dim),
            nn.Sigmoid(),
        )

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def forward(self, x):
        h = self.encoder(x)
        mu = self.fc_mu(h)
        logvar = self.fc_logvar(h)

        z = self.reparameterize(mu, logvar)
        x_recon = self.decoder(z)

        kl_loss = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
        recon_loss = F.mse_loss(x_recon, x, reduction="sum")
        total_loss = recon_loss + kl_loss

        return z, total_loss, x_recon


class DeepADR_MLP(nn.Module):
    def __init__(
        self,
        input_dim1,
        input_dim2,
        cnn_output_dim,
        vae_latent_dim,
        mlp_hidden_dim,
        output_dim,
    ):
        super(DeepADR_MLP, self).__init__()

        self.mol_feature_dim = input_dim1
        self.adr_feature_dim = input_dim1

        self.attention = AttentionMechanism(drug_dim=input_dim1, adr_dim=input_dim1)
        attention_output_dim = input_dim1 * 2

        self.cnn_branch = CNNBranch(
            input_dim=attention_output_dim, output_dim=cnn_output_dim
        )
        self.vae_branch = VAEBranch(input_dim=input_dim2, latent_dim=vae_latent_dim)

        combined_input_dim = cnn_output_dim + vae_latent_dim
        self.mlp = nn.Sequential(
            nn.Linear(combined_input_dim, mlp_hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(mlp_hidden_dim, mlp_hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(mlp_hidden_dim // 2, output_dim),
        )

    def forward(self, mol_features, tar_features, adr_features):
        attention_output = self.attention(mol_features, adr_features)
        cnn_out = self.cnn_branch(attention_output)

        vae_out, vae_loss, _ = self.vae_branch(tar_features)

        combined_features = torch.cat([cnn_out, vae_out], dim=1)
        final_out = self.mlp(combined_features)

        return final_out, vae_loss


class RMSELoss(nn.Module):
    def __init__(self):
        super(RMSELoss, self).__init__()
        self.mse = nn.MSELoss()

    def forward(self, y_pred, y_true):
        return torch.sqrt(self.mse(y_pred, y_true))


class MAELoss(nn.Module):
    def __init__(self):
        super(MAELoss, self).__init__()

    def forward(self, y_pred, y_true):
        return torch.mean(torch.abs(y_pred - y_true))


def train_model(model, train_loader, val_loader, criterion, optimizer, device, num_epochs=10):
    mae_criterion = MAELoss()

    for epoch in range(num_epochs):
        model.train()
        train_rmse = 0.0
        train_mae = 0.0
        train_preds = []
        train_targets = []

        for mol_features, tar_features, adr_features, target in train_loader:
            mol_features = mol_features.to(device)
            tar_features = tar_features.to(device)
            adr_features = adr_features.to(device)
            target = target.to(device)

            optimizer.zero_grad()
            output, vae_loss = model(mol_features, tar_features, adr_features)

            regression_loss = criterion(output.squeeze(), target)
            total_loss = regression_loss + 0.1 * vae_loss

            total_loss.backward()
            optimizer.step()

            train_rmse += regression_loss.item()
            train_mae += mae_criterion(output.squeeze(), target).item()
            train_preds.extend(output.squeeze().detach().cpu().numpy())
            train_targets.extend(target.cpu().numpy())

        model.eval()
        val_rmse = 0.0
        val_mae = 0.0
        val_preds = []
        val_targets = []

        with torch.no_grad():
            for mol_features, tar_features, adr_features, target in val_loader:
                mol_features = mol_features.to(device)
                tar_features = tar_features.to(device)
                adr_features = adr_features.to(device)
                target = target.to(device)

                output, vae_loss = model(mol_features, tar_features, adr_features)
                regression_loss = criterion(output.squeeze(), target)
                total_loss = regression_loss + 0.1 * vae_loss

                val_rmse += regression_loss.item()
                val_mae += mae_criterion(output.squeeze(), target).item()
                val_preds.extend(output.squeeze().cpu().numpy())
                val_targets.extend(target.cpu().numpy())

        avg_train_rmse = train_rmse / len(train_loader)
        avg_train_mae = train_mae / len(train_loader)
        avg_val_rmse = val_rmse / len(val_loader)
        avg_val_mae = val_mae / len(val_loader)

        train_pcc, _ = pearsonr(train_targets, train_preds)
        val_pcc, _ = pearsonr(val_targets, val_preds)

        print(f"Epoch {epoch + 1}/{num_epochs}")
        print(
            f"Train - RMSE: {avg_train_rmse:.4f}, MAE: {avg_train_mae:.4f}, PCC: {train_pcc:.4f}"
        )
        print(
            f"Val   - RMSE: {avg_val_rmse:.4f}, MAE: {avg_val_mae:.4f}, PCC: {val_pcc:.4f}"
        )
        print("-" * 50)


def evaluate_model(model, test_loader, criterion, device):
    mae_criterion = MAELoss()

    model.eval()
    test_rmse = 0.0
    test_mae = 0.0
    test_preds = []
    test_targets = []

    with torch.no_grad():
        for mol_features, tar_features, adr_features, target in test_loader:
            mol_features = mol_features.to(device)
            tar_features = tar_features.to(device)
            adr_features = adr_features.to(device)
            target = target.to(device)

            output, _ = model(mol_features, tar_features, adr_features)

            test_rmse += criterion(output.squeeze(), target).item()
            test_mae += mae_criterion(output.squeeze(), target).item()
            test_preds.extend(output.squeeze().cpu().numpy())
            test_targets.extend(target.cpu().numpy())

    avg_test_rmse = test_rmse / len(test_loader)
    avg_test_mae = test_mae / len(test_loader)
    test_pcc, _ = pearsonr(test_targets, test_preds)

    print("Test Results:")
    print(f"RMSE: {avg_test_rmse:.4f}")
    print(f"MAE:  {avg_test_mae:.4f}")
    print(f"PCC:  {test_pcc:.4f}")
