#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import torch
import torch.nn as nn
import torch.nn.functional as F


# In[ ]:


# Global Context Block (GCB)
class GlobalContextBlock(nn.Module):
    def __init__(self, channels, reduction=16):
        super(GlobalContextBlock, self).__init__()
        self.conv_mask = nn.Conv2d(channels, 1, kernel_size=1)
        self.softmax = nn.Softmax(dim=2)

        self.transform = nn.Sequential(
            nn.Conv2d(channels, channels // reduction, kernel_size=1),
            nn.LayerNorm([channels // reduction, 1, 1]),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels // reduction, channels, kernel_size=1)
        )

    def forward(self, x):
        batch, c, h, w = x.size()
        input_x = x

        # context modeling: spatial pooling
        context_mask = self.conv_mask(x).view(batch, 1, -1)
        context_mask = self.softmax(context_mask)
        x_context = x.view(batch, c, -1)
        context = torch.bmm(x_context, context_mask.permute(0, 2, 1)).view(batch, c, 1, 1)

        # transform and add
        context = self.transform(context)
        out = input_x + context
        return out


# In[ ]:


import torch_dct as dct  # you need to install torch-dct

class FrequencyChannelAttention(nn.Module):
    def __init__(self, channels):
        super(FrequencyChannelAttention, self).__init__()
        self.fc = nn.Sequential(
            nn.Linear(channels, channels // 8),
            nn.ReLU(inplace=True),
            nn.Linear(channels // 8, channels),
            nn.Sigmoid()
        )

    def forward(self, x):
        # Apply DCT along spatial dimensions
        x_freq = dct.dct_2d(x)  # returns same shape as input
        freq_avg = x_freq.mean(dim=(2, 3))  # [B, C]

        weights = self.fc(freq_avg).unsqueeze(-1).unsqueeze(-1)
        return x * weights


# In[ ]:


class ASPP(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(ASPP, self).__init__()
        self.atrous_block1 = nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=1, padding=0, dilation=1)
        self.atrous_block3 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=3, dilation=3)
        self.atrous_block6 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=6, dilation=6)
        self.atrous_block12 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=12, dilation=12)
        self.image_pool = nn.AdaptiveAvgPool2d(1)
        self.image_conv = nn.Conv2d(in_channels, out_channels, kernel_size=1)

        self.conv_1x1_output = nn.Conv2d(out_channels * 5, out_channels, kernel_size=1)
        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        size = x.shape[2:]

        image_features = self.image_pool(x)
        image_features = self.image_conv(image_features)
        image_features = F.interpolate(image_features, size=size, mode='bilinear', align_corners=True)

        x1 = self.atrous_block1(x)
        x2 = self.atrous_block3(x)
        x3 = self.atrous_block6(x)
        x4 = self.atrous_block12(x)

        x = torch.cat([x1, x2, x3, x4, image_features], dim=1)
        x = self.conv_1x1_output(x)
        x = self.bn(x)
        x = self.relu(x)
        return x


# In[ ]:


import torch
import torch.nn as nn
import torch.nn.functional as F

class HCSFBlock(nn.Module):
    def __init__(self, in_low=None, in_mid=None, in_high=None, out_channels=None):
        super(HCSFBlock, self).__init__()
        
        # Store channel information
        self.in_mid = in_mid
        self.out_channels = out_channels if out_channels is not None else in_mid
        
        # Determine which connections to use
        self.use_low = in_low is not None
        self.use_high = in_high is not None
        
        # Low-resolution feature processing
        if self.use_low:
            self.conv_low = nn.Sequential(
                nn.Conv2d(in_low, self.out_channels, kernel_size=1),
                nn.BatchNorm2d(self.out_channels),
                nn.ReLU(inplace=True)
            )
            self.gate_low = nn.Sequential(
                nn.Conv2d(self.in_mid, 1, kernel_size=1),
                nn.Sigmoid()
            )
        
        # High-resolution feature processing
        if self.use_high:
            self.conv_high = nn.Sequential(
                nn.Conv2d(in_high, self.out_channels, kernel_size=1),
                nn.BatchNorm2d(self.out_channels),
                nn.ReLU(inplace=True)
            )
            self.gate_high = nn.Sequential(
                nn.Conv2d(self.in_mid, 1, kernel_size=1),
                nn.Sigmoid()
            )
        
        # If output channels differ from input mid channels, add projection
        if self.out_channels != self.in_mid:
            self.proj_mid = nn.Sequential(
                nn.Conv2d(self.in_mid, self.out_channels, kernel_size=1),
                nn.BatchNorm2d(self.out_channels),
                nn.ReLU(inplace=True)
            )
        else:
            self.proj_mid = nn.Identity()
    
    def forward(self, f_mid, f_low=None, f_high=None):
        # Project middle features if needed
        fused = self.proj_mid(f_mid)
        
        # Process low-resolution features
        if self.use_low and f_low is not None:
            # Downsample low features to match mid spatial dimensions
            f_low_down = F.adaptive_avg_pool2d(f_low, f_mid.shape[2:])
            f_low_proj = self.conv_low(f_low_down)
            gate_l = self.gate_low(f_mid)
            fused = fused + gate_l * f_low_proj
        
        # Process high-resolution features
        if self.use_high and f_high is not None:
            # Upsample high features to match mid spatial dimensions
            f_high_up = F.interpolate(f_high, size=f_mid.shape[2:], mode='bilinear', align_corners=False)
            f_high_proj = self.conv_high(f_high_up)
            gate_h = self.gate_high(f_mid)
            fused = fused + gate_h * f_high_proj
        
        return fused


# In[ ]:


class MPARBlock(nn.Module):
    def __init__(self, channels):
        super(MPARBlock, self).__init__()

        # Channel attention
        self.ca = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(channels, channels // 8, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels // 8, channels, 1),
            nn.Sigmoid()
        )

        # Spatial attention with large kernel conv
        self.sa = nn.Sequential(
            nn.Conv2d(channels, 1, kernel_size=9, padding=4),
            nn.Sigmoid()
        )

    
        self.fa = FrequencyChannelAttention(channels)
     
        self.ga = GlobalContextBlock(channels)
        # Learnable fusion weights
        self.alpha = nn.Parameter(torch.ones(4))

        # Final projection
        self.out_conv = nn.Conv2d(channels, channels, kernel_size=1)

    def forward(self, x):
        ac = self.ca(x) * x
        as_ = self.sa(x) * x
       
        af = self.fa(x) 
        ag = self.ga(x) 
        weights = F.softmax(self.alpha, dim=0)
        fused = weights[0]*ac + weights[1]*as_ + weights[2]*af + weights[3]*ag

        return self.out_conv(fused)



# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:




