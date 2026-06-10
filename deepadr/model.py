import torch
import torch.nn as nn
import torch.nn.functional as F
from kan import KAN


class AttentionMechanism(nn.Module):
    def __init__(self, drug_dim: int, adr_dim: int, hidden_dim: int = 128):
        super().__init__()
        self.drug_projection = nn.Linear(drug_dim, hidden_dim)
        self.adr_projection = nn.Linear(adr_dim, hidden_dim)
        self.attention = nn.Linear(hidden_dim, 1)

    def forward(self, drug_features: torch.Tensor, adr_features: torch.Tensor) -> torch.Tensor:
        energy = torch.tanh(
            self.drug_projection(drug_features) + self.adr_projection(adr_features)
        )
        attention_scores = torch.sigmoid(self.attention(energy))
        attended_drug = drug_features * attention_scores
        return torch.cat([attended_drug, adr_features], dim=1)


class CNNBranch(nn.Module):
    def __init__(self, input_dim: int, output_dim: int, dropout: float = 0.2):
        super().__init__()
        self.bn1 = nn.BatchNorm1d(1)
        self.conv1 = nn.Conv1d(1, 16, kernel_size=3, stride=1, padding=1)
        self.bn2 = nn.BatchNorm1d(16)
        self.conv2 = nn.Conv1d(16, 32, kernel_size=3, stride=1, padding=1)
        self.bn3 = nn.BatchNorm1d(32)
        self.pool = nn.MaxPool1d(kernel_size=2, stride=2)
        self.dropout = nn.Dropout(dropout)

        fc_input_dim = (input_dim // 4) * 32
        self.fc1 = nn.Linear(fc_input_dim, 512)
        self.bn4 = nn.BatchNorm1d(512)
        self.fc2 = nn.Linear(512, output_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch_size = x.size(0)
        x = x.view(batch_size, 1, -1)
        x = self.bn1(x)
        x = self.pool(F.relu(self.bn2(self.conv1(x))))
        x = self.dropout(x)
        x = self.pool(F.relu(self.bn3(self.conv2(x))))
        x = self.dropout(x)
        x = x.view(batch_size, -1)
        x = self.dropout(F.relu(self.bn4(self.fc1(x))))
        return self.fc2(x)


class VAEBranch(nn.Module):
    def __init__(self, input_dim: int, latent_dim: int, dropout: float = 0.2):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(512, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(dropout),
        )
        self.fc_mu = nn.Linear(128, latent_dim)
        self.fc_logvar = nn.Linear(128, latent_dim)

    def reparameterize(self, mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        if not self.training:
            return mu
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        hidden = self.encoder(x)
        mu = self.fc_mu(hidden)
        logvar = self.fc_logvar(hidden)
        return self.reparameterize(mu, logvar)


class DeepADR(nn.Module):
    """DeepADR model for both ADR occurrence classification and frequency regression."""

    def __init__(
        self,
        mol_dim: int,
        target_dim: int,
        adr_dim: int,
        cnn_output_dim: int = 128,
        vae_latent_dim: int = 128,
        kan_hidden_dim: int = 16,
        output_dim: int = 1,
        kan_grid: int = 2,
        kan_k: int = 3,
        kan_device: str = "cpu",
        dropout: float = 0.2,
    ):
        super().__init__()
        self.attention = AttentionMechanism(drug_dim=mol_dim, adr_dim=adr_dim)
        self.cnn_branch = CNNBranch(
            input_dim=mol_dim + adr_dim,
            output_dim=cnn_output_dim,
            dropout=dropout,
        )
        self.vae_branch = VAEBranch(
            input_dim=target_dim,
            latent_dim=vae_latent_dim,
            dropout=dropout,
        )
        self.kan_layer = KAN(
            width=[cnn_output_dim + vae_latent_dim, kan_hidden_dim, output_dim],
            grid=kan_grid,
            k=kan_k,
            seed=42,
            device=kan_device,
        )

    def forward(
        self,
        mol_features: torch.Tensor,
        target_features: torch.Tensor,
        adr_features: torch.Tensor,
    ) -> torch.Tensor:
        attention_output = self.attention(mol_features, adr_features)
        cnn_output = self.cnn_branch(attention_output)
        vae_output = self.vae_branch(target_features)
        combined = torch.cat([cnn_output, vae_output], dim=1)
        return self.kan_layer(combined).view(-1)
