#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import torch
import torch.nn as nn
from .encoder import EfficientNetB0Encoder
from .decoder import UNetDecoder


# In[ ]:


class FullSegmentationModel(nn.Module):
    def __init__(self, encoder, decoder):
        super(FullSegmentationModel, self).__init__()
        self.encoder = encoder  # ResNet18 or similar
        self.decoder = decoder

    def forward(self, x):
        
        f1, f2, f3, f4 = self.encoder(x)
        return self.decoder(f1, f2, f3, f4)


# In[ ]:


def create_model():
        model = FullSegmentationModel(
            encoder=EfficientNetB0Encoder(),
            decoder=UNetDecoder([16, 24, 40, 80])
        )
        
        pretrained_encoder_path = #path of the encoder trained from stage 1 of the model
        if os.path.exists(pretrained_encoder_path):
            model.encoder = EfficientNetB0Encoder(pretrained_path=pretrained_encoder_path)
        
        return model
    
        # First run (or continue from where you left off)
        results = train_kfold_cross_validation(
        model_class=create_model,
        n_folds=5,
        total_epochs=100,
        resume=False
        )

