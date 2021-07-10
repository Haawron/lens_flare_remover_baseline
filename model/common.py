import torch.nn as nn
import torch


class ConvolutionBlock(nn.Module):

    def __init__(self, dim=16, activation=True):
        super().__init__()
        layers = [
            nn.Conv2d(dim, dim, kernel_size=3, padding=1),  # padding="same"
            nn.BatchNorm2d(dim),
        ]
        if activation:
            layers.append(nn.LeakyReLU(0.1, inplace=False))
        self.block = nn.Sequential(*layers)

    def forward(self, x):
        return self.block(x)


class ResidualBlock(nn.Module):

    def __init__(self, dim=16):
        super().__init__()
        layers = [
            nn.LeakyReLU(0.1, inplace=False),
            nn.BatchNorm2d(dim),
            ConvolutionBlock(dim),
            ConvolutionBlock(dim, False),
        ]
        self.block = nn.Sequential(*layers)
        self.bn = nn.BatchNorm2d(dim)

    def forward(self, x):
        x1 = self.block(x)
        x = self.bn(x)
        x += x1
        return x


class ResidualBlock2(nn.Module):

    def __init__(self, in_dim=16, out_dim=16, dropout_rate=.1):
        super().__init__()
        layers = [
            nn.Dropout(dropout_rate, inplace=True),
            nn.Conv2d(in_dim, out_dim, kernel_size=3, padding=1),
            ResidualBlock(out_dim),
            ResidualBlock(out_dim),
            nn.LeakyReLU(0.1, inplace=False),
        ]
        self.block = nn.Sequential(*layers)

    def forward(self, x):
        return self.block(x)


class Up(nn.Module):

    def __init__(self, in_dim=16, skip_dim=16, concat=True):
        super().__init__()
        self.up = nn.ConvTranspose2d(
            in_dim, in_dim // 2, kernel_size=3, stride=2, padding=1, output_padding=1)
        self.res = ResidualBlock2(in_dim//2 + skip_dim if concat else in_dim//2, in_dim//2)

    def forward(self, x, x_skip=None):
        x = self.up(x)  # C = in_dim//2
        if x_skip is not None:
            x = torch.cat([x, x_skip], dim=1)  # C = in_dim
        x = self.res(x)  # C = in_dim//2
        return x

