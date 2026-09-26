"""DCGAN generator / discriminator used by the DFMS-HL model-extraction attack.

Architectures are copied verbatim from the reference implementation
(val-iisc/Hard-Label-Model-Stealing, dcgan_model.py) so that a generator
trained here has the same capacity and inductive bias as the paper's:

  Generator      z(100) -> 4x4 -> 8x8 -> 16x16 -> 32x32, tanh output in [-1, 1]
  Discriminator  32x32 -> 16 -> 8 -> 4 -> 2 -> 1, sigmoid output

Both work on images in the GAN's own [-1, 1] space; the conversion to the
victim's normalisation is done by the attack code, not here.
"""
import torch.nn as nn


class Generator(nn.Module):
    def __init__(self, nc=3, nz=100, ngf=64):
        super().__init__()
        self.nz = nz
        self.main = nn.Sequential(
            nn.ConvTranspose2d(nz, ngf * 8, 4, 1, 0, bias=False),
            nn.BatchNorm2d(ngf * 8),
            nn.ReLU(True),

            nn.ConvTranspose2d(ngf * 8, ngf * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf * 4),
            nn.ReLU(True),

            nn.ConvTranspose2d(ngf * 4, ngf * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf * 2),
            nn.ReLU(True),

            nn.ConvTranspose2d(ngf * 2, ngf, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf),
            nn.ReLU(True),

            nn.ConvTranspose2d(ngf, nc, kernel_size=1, stride=1, padding=0, bias=False),
            nn.Tanh(),
        )

    def forward(self, z):
        return self.main(z)


class Discriminator(nn.Module):
    def __init__(self, nc=3, ndf=64):
        super().__init__()
        self.main = nn.Sequential(
            nn.Conv2d(nc, ndf, 4, 2, 1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(ndf, ndf * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf * 2),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(ndf * 2, ndf * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf * 4),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(ndf * 4, ndf * 8, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf * 8),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(ndf * 8, 1, 2, 2, 0, bias=False),
            nn.Sigmoid(),
        )

    def forward(self, x):
        return self.main(x).view(-1)


def weights_init(m):
    """DCGAN initialisation from the reference code: N(0, 0.02) for conv
    weights, N(1, 0.02) / 0 for BatchNorm scale / shift."""
    name = m.__class__.__name__
    if name.find("Conv") != -1:
        m.weight.data.normal_(0.0, 0.02)
    elif name.find("BatchNorm") != -1:
        m.weight.data.normal_(1.0, 0.02)
        m.bias.data.fill_(0)
