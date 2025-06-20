import torch.nn as nn

class Generator3D(nn.Module):

    def __init__(self, in_channels=1, out_channels=1, features=32):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Conv3d(in_channels, features, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm3d(features),
            nn.ReLU(inplace=True),

            nn.Conv3d(features, features*2, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm3d(features*2),
            nn.ReLU(inplace=True),
        )
        self.decoder = nn.Sequential(
            nn.ConvTranspose3d(features*2, features, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm3d(features),
            nn.ReLU(inplace=True),

            nn.ConvTranspose3d(features, out_channels, kernel_size=4, stride=2, padding=1),
            nn.Tanh(),
        )

    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        return x
