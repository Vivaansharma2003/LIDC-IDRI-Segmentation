import torch
import torch.nn as nn
import torch.nn.functional as F
from seprable_unet3D_parts import *
class Unet3D(nn.Module):
    def __init__(self,n_class=1):
        super(Unet3D,self).__init__()
        self.inp=nn.Conv3d(3,64,kernal_size=3,padding=True)
        self.part2=Down3D(64,128,expand=1)
        self.part3=Down3D(128,256,expand=1)
        self.part4=Down3D(256,512,expand=1)
        self.part5=Up3D(512,256,expand=1)
        self.part6=Up3D(256,128,expand=1)
        self.part7=Up3D(128,64,expand=1)
        self.out=nn.Conv3d(64,n_class,1)
        self.maxpool=nn.MaxPool3d(2)
    def forward(self,x):
        x1_in = self.inp(x)
        x1 = self.maxpool(x1_in)
        x2_in = self.part2(x1)
        x2 = self.maxpool(x2_in)
        x3_in = self.part3(x2)
        x3 = self.maxpool(x3_in)
        x4_in = self.part4(x3)
        x4 = self.maxpool(x4_in)
        x5 = self.part5(x4)

        x6 = self.part5(x5, x4_in)
        x7 = self.part6(x6, x3_in)
        x8 = self.part7(x7, x2_in)
        
        out= self.out(x8)
        return F.sigmoid(out)
        