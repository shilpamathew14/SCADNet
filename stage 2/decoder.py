#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import torch
import torch.nn as nn
from .blocks import HCSFBlock, MPARBlock, ASPP, GlobalContextBlock, FrequencyChannelAttention


# In[ ]:


class UNetDecoder(nn.Module):
    def __init__(self, channels):
        super(UNetDecoder, self).__init__()

        # HCSF block initialization with correct channel mapping
        self.hcsf3 = HCSFBlock(in_low=None, in_mid=channels[3], in_high=channels[2], out_channels=channels[3]) # 2 input 
        self.hcsf2 = HCSFBlock(in_low=channels[1], in_mid=channels[2], in_high=channels[3], out_channels=channels[2])
        self.hcsf1 = HCSFBlock(in_low=channels[0], in_mid=channels[1], in_high=channels[2], out_channels=channels[1])

        self.mpar3 = MPARBlock(channels[3])
        self.mpar2 = MPARBlock(channels[2])
        self.mpar1 = MPARBlock(channels[1])

        self.up3 = nn.ConvTranspose2d(channels[3], channels[2], kernel_size=2, stride=2)
        self.up2 = nn.ConvTranspose2d(channels[2], channels[1], kernel_size=2, stride=2)
        self.up1 = nn.ConvTranspose2d(channels[1], channels[0], kernel_size=2, stride=2)
        self.up0 = nn.ConvTranspose2d(channels[0], channels[0], kernel_size=2, stride=2)
        
        self.final_conv = nn.Sequential(
            nn.Conv2d(channels[0], 5, kernel_size=1)
        )

    def forward(self, f1, f2, f3, f4):

        d3 = self.hcsf3(f4, f_low=None, f_high=f3) 
        d3 = self.mpar3(d3)
        d3 = self.up3(d3)


       
        d2 = self.hcsf2(d3, f_low=f2, f_high=f4)  
        d2 = self.mpar2(d2)
        d2 = self.up2(d2)


        d1 = self.hcsf1(d2, f_low=f1, f_high=f3) 
        d1 = self.mpar1(d1)
        d1 = self.up1(d1)


        d0 = self.up0(d1) 
        out = self.final_conv(d0)


        return out


# In[ ]:




