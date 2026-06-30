"""
Compact U-Net generator: RGB (3ch) -> NIR (1ch). Imported only when torch is
present (the `neural` Poetry group). Kept small so it trains in a couple of hours
on a single 24 GB GPU and runs tiled at full resolution.
"""

import torch
import torch.nn as nn


def _block(cin, cout):
    return nn.Sequential(
        nn.Conv2d(cin, cout, 3, padding=1), nn.BatchNorm2d(cout), nn.ReLU(inplace=True),
        nn.Conv2d(cout, cout, 3, padding=1), nn.BatchNorm2d(cout), nn.ReLU(inplace=True),
    )


class CompactUNet(nn.Module):
    """4-level U-Net, base 48 channels, sigmoid NIR output in [0,1]."""

    def __init__(self, in_ch=3, out_ch=1, base=48):
        super().__init__()
        b = base
        self.d1 = _block(in_ch, b)
        self.d2 = _block(b, b * 2)
        self.d3 = _block(b * 2, b * 4)
        self.d4 = _block(b * 4, b * 8)
        self.pool = nn.MaxPool2d(2)
        self.bott = _block(b * 8, b * 16)

        self.up4 = nn.ConvTranspose2d(b * 16, b * 8, 2, stride=2)
        self.u4 = _block(b * 16, b * 8)
        self.up3 = nn.ConvTranspose2d(b * 8, b * 4, 2, stride=2)
        self.u3 = _block(b * 8, b * 4)
        self.up2 = nn.ConvTranspose2d(b * 4, b * 2, 2, stride=2)
        self.u2 = _block(b * 4, b * 2)
        self.up1 = nn.ConvTranspose2d(b * 2, b, 2, stride=2)
        self.u1 = _block(b * 2, b)
        self.out = nn.Conv2d(b, out_ch, 1)

    def forward(self, x):
        c1 = self.d1(x)
        c2 = self.d2(self.pool(c1))
        c3 = self.d3(self.pool(c2))
        c4 = self.d4(self.pool(c3))
        bn = self.bott(self.pool(c4))
        x = self.u4(torch.cat([self.up4(bn), c4], 1))
        x = self.u3(torch.cat([self.up3(x), c3], 1))
        x = self.u2(torch.cat([self.up2(x), c2], 1))
        x = self.u1(torch.cat([self.up1(x), c1], 1))
        return torch.sigmoid(self.out(x))
