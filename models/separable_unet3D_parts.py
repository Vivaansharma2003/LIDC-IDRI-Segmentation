import torch
import torch.nn as nn
import torch.nn.functional as F

def fixed_padding_3D(inputs,kernel_size,dilation):
    kernal_size_effective=kernel_size+(kernel_size-1)*(dilation-1)
    pad_total=kernal_size_effective-1
    pad_beg=pad_total//2
    pad_end=pad_total-pad_beg
    padded_inputs=F.pad(inputs,(pad_beg,pad_end,pad_beg,pad_end,pad_beg,pad_end))
    return padded_inputs
class InvertedResidual3D(nn.Module):
    def __init__(self,inp,oup,expand):
        super(InvertedResidual3D,self).__init__()
        self.expand=expand
        self.conv=nn.Sequential(
            nn.Conv3d(inp,inp,3,1,0,dilation=expand,groups=inp,bias=False),
            nn.BatchNorm3d(inp),
            nn.ReLU6(inplace=True),
            nn.Conv3d(inp,oup,1,1,0,1,bias=False)

        )
    def forward(self,x):
        x_pad=fixed_padding_3D(x,3,self.expand)
        y=self.conv(x_pad)
        return y
class Down3D(nn.Module):
    def __init__(self, inp_channel, out_channel, expand):
        super(Down3D, self).__init__()
        self.deepwise1 = InvertedResidual3D(inp_channel, inp_channel, expand)
        self.deepwise2 = InvertedResidual3D(inp_channel, out_channel, expand)
        self.resnet = nn.Conv3d(inp_channel, out_channel, kernel_size=1, stride=1, padding=0, bias=False)

    def forward(self, input):
        resnet = self.resnet(input)
        x = self.deepwise1(input)
        x = self.deepwise2(x)
        out = torch.add(resnet, x)
        return out

class Up3D(nn.Module):
    def __init__(self, inp_channel, out_channel, expand):
        super(Up3D, self).__init__()
        self.up = nn.ConvTranspose3d(inp_channel, out_channel, kernel_size=2, stride=2)
        self.deepwise1 = InvertedResidual3D(inp_channel, inp_channel, expand)
        self.deepwise2 = InvertedResidual3D(inp_channel, out_channel, expand)
        self.resnet = nn.Conv3d(inp_channel, out_channel, kernel_size=1, stride=1, padding=0, bias=False)

    def forward(self, x, y):
        x = self.up(x)
        x1 = torch.cat([x, y], dim=1)
        x = self.deepwise1(x1)
        x = self.deepwise2(x)
        resnet = self.resnet(x1)
        out = torch.add(resnet, x)
        return out
