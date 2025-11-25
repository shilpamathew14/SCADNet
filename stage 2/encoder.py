#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import torch.nn as nn
from torchvision.models import efficientnet_b0, EfficientNet_B0_Weights

class EfficientNetB0Encoder(nn.Module):
    def __init__(self, pretrained_path=None):
        super(EfficientNetB0Encoder, self).__init__()

        model = efficientnet_b0(weights=None)  

        if pretrained_path:
            checkpoint = torch.load(pretrained_path, map_location='cpu')
            state_dict = checkpoint.get('state_dict', checkpoint)
            new_state_dict = {k.replace("model.", "").replace("encoder.", ""): v for k, v in state_dict.items()}
            model.load_state_dict(new_state_dict, strict=False)

        # EfficientNet feature extractor stages
        self.stem = model.features[0]                     
        self.block1 = model.features[1]                   
        self.block2 = model.features[2]                  
        self.block3 = model.features[3]                  
        self.block4 = model.features[4]                   
        self.block5 = model.features[5]                  
        self.aspp = ASPP(in_channels=80, out_channels=80) 

    
    def forward(self, x):
        x = self.stem(x)
        f1 = self.block1(x)      
        f2 = self.block2(f1)     
        f3 = self.block3(f2)      
        f4 = self.block4(f3)      
        # Apply ASPP to the deepest feature
        f4_enhanced = self.aspp(f4)  
        
        return f1, f2, f3, f4_enhanced

