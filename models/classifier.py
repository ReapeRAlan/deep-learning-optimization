import torch
from torch import nn

class Classifier(nn.Module):
    def __init__(self, latent_dim, output_dim):
        super(Classifier, self).__init__()
        self.fc = nn.Sequential(
            nn.Linear(latent_dim, 128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, output_dim),
        )

    def forward(self, x):
        return self.fc(x)
