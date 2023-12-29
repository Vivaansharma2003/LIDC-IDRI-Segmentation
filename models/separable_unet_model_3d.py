import torch
import torch.nn as nn
import torch.nn.functional as F
from .separable_unet3D_parts import *
class Separable_Unet_3D(nn.Module):
    def __init__(self, class_num):
        super(Separable_Unet_3D, self).__init__()
        self.inp = nn.Conv3d(3, 64, kernel_size=3, padding=1)
        self.part2 = Down3D(64, 128, expand=1)
        self.part3 = Down3D(128, 256, expand=2)
        self.part4 = Down3D(256, 512, expand=3)
        self.part5 = Up3D(512, 256, expand=1)
        self.part6 = Up3D(256, 128, expand=1)
        self.part7 = Up3D(128, 64, expand=1)
        self.out = nn.Conv3d(64, class_num, kernel_size=1)
        self.maxpool = nn.MaxPool3d(2)

    def forward(self, x):
        x1_in = self.inp(x)
        x1 = self.maxpool(x1_in)
        x2_in = self.part2(x1)
        x2 = self.maxpool(x2_in)
        x3_in = self.part3(x2)
        x3 = self.maxpool(x3_in)
        x4 = self.maxpool(x3)
        x5 = self.part5(x4,x3_in)
        x6 = self.part5(x5, x2_in)
        x7 = self.part6(x6, x1_in)
        out = self.out(x)
        return torch.sigmoid(out)
