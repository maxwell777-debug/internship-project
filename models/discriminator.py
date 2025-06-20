import torch
import torch.nn as nn

class Discriminator3D(nn.Module):
    
    def __init__(self, in_channels=3, features=32):
        super().__init__()
        self.model = nn.Sequential(
            nn.Conv3d(in_channels, features, 4, 2, 1),
            nn.LeakyReLU(0.2, inplace=True),
            
            nn.Conv3d(features, features*2, 4, 2, 1),
            nn.BatchNorm3d(features*2),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv3d(features*2, 1, 4, 1, 1),
            nn.Sigmoid()
        )

    def forward(self, x, y):
        return self.model(torch.cat([x, y], dim=1))

