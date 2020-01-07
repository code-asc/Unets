import torch
import torch.nn as nn
import torch.nn.functional as F
from unet.convutils import TwoLayerConv
from unet.convutils import DownCastConv
from unet.convutils import UpCastConv
from unet.convutils import LastLayerConv
from unet.convutilslite import TwoLayerConvLite
from unet.convutilslite import DownCastConvLite
from unet.convutilslite import UpCastConvLite
from unet.convutilslite import LastLayerConvLite

class UnetLite(nn.Module):
    def __init__(self, nchannels, nclasses):
        super(UnetLite, self).__init__()

        self.conv = TwoLayerConvLite(nchannels, 64)
        self.down1 = DownCastConvLite(64, 128)
        self.down2 = DownCastConvLite(128, 128)

        self.up1 = UpCastConvLite(256, 64)
        self.up2 = UpCastConvLite(128, 64)


        self.last =LastLayerConvLite(64, nclasses)

    def forward(self, x):
        x1 = self.conv(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)

        x = self.up1(x3, x2)
        x = self.up2(x, x1)

        return self.last(x)


class Unet(nn.Module):
    def __init__(self, nchannels, nclasses):
        super(Unet, self).__init__()

        self.conv = TwoLayerConv(nchannels, 64)
        self.down1 = DownCastConv(64, 128)
        self.down2 = DownCastConv(128, 256)
        self.down3 = DownCastConv(256, 512)
        self.down4 = DownCastConv(512, 1024)
        self.down5 = DownCastConv(1024, 1024)

        self.up1 = UpCastConv(2048, 512)
        self.up2 = UpCastConv(1024, 256)
        self.up3 = UpCastConv(512, 128)
        self.up4 = UpCastConv(256, 64)
        self.up5 = UpCastConv(128, 64)


        self.last =LastLayerConv(64, nclasses)

    def forward(self, x):
        x1 = self.conv(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)
        x6 = self.down4(x5)

        x = self.up1(x6, x5)
        x = self.up2(x5, x4)
        x = self.up3(x, x3)
        x = self.up4(x, x2)
        x = self.up5(x, x1)

        return self.last(x)
