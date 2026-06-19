import torch
import torch.nn as nn

class ResNetMLP(nn.Module):
    def __init__(self, input_dim=3072, hidden_dim=512, dropout_rate=0.2, noise_level=0.0):
        super().__init__()
        self.noise_level = noise_level
        self.input_layer = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout_rate)
        )
        
        # Residual Block
        self.res_block = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout_rate),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout_rate)
        )
        
        self.output_layer = nn.Linear(hidden_dim, 1)

    def forward(self, x):
        if self.training and self.noise_level > 0.0:
            data_std = x.std()
            noise_std = data_std * self.noise_level
            x = x + torch.randn_like(x) * noise_std
        h = self.input_layer(x)
        h = h + self.res_block(h)
        return self.output_layer(h).squeeze(-1)