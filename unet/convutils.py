import torch
import torch.nn as nn
import torch.nn.functional as F

class TwoLayerConv(nn.Module):
    def __init__(self, inChannel, outChannel):
        super().__init__()
        self.two_layer_conv = nn.Sequential(nn.Conv2d(inChannel, outChannel, kernel_size=3, padding=1),
                                            nn.BatchNorm2d(outChannel),
                                            nn.ReLU(),
                                            nn.Conv2d(outChannel, outChannel, kernel_size=3, padding=1),
                                            nn.BatchNorm2d(outChannel),
                                            nn.ReLU())

    def forward(self, x):
        return self.two_layer_conv(x)


class DownCastConv(nn.Module):
    def __init__(self, inChannel, outChannel):
        super().__init__()
        self.down_cast_conv = nn.Sequential(nn.MaxPool2d(2),
                                            TwoLayerConv(inChannel, outChannel))

    def forward(self, x):
        return self.down_cast_conv(x)


class UpCastConv(nn.Module):
    def __init__(self, inChannel, outChannel):
        super().__init__()
        self.inChannel = inChannel
        self.outChannel = outChannel
        self.up_cast_conv = nn.Sequential(nn.ConvTranspose2d(inChannel//2, inChannel//2, kernel_size=2, stride=2))
        self.conv = TwoLayerConv(inChannel, outChannel)


    def forward(self, x_1, x_2):
        x_1 = self.up_cast_conv(x_1)
        diff_vertical = x_2.size()[2] - x_1.size()[2]
        diff_horizontal = x_2.size()[3] - x_1.size()[3]

        x_1 = F.pad(x_1, (diff_horizontal//2, diff_horizontal - diff_horizontal//2,
                          diff_vertical//2, diff_vertical - diff_vertical//2))

        temp = torch.cat((x_2, x_1),dim=1)
        return self.conv(temp)

class LastLayerConv(nn.Module):
    def __init__(self, inChannel, outChannel):
        super(LastLayerConv, self).__init__()
        self.conv = nn.Conv2d(inChannel, outChannel, kernel_size=1)

    def forward(self, x):
        return self.conv(x)
