import torch
import torch.nn as nn
import torchvision.models as models

#pre-trained model as feature extractor
class CatDogModel(nn.Module):
    def __init__(self, num_classes=2, freeze_features=True):
        super().__init__()
        
        self.model = models.mobilenet_v2(weights=models.MobileNet_V2_Weights.DEFAULT)
        
        if freeze_features:
            for param in self.model.features.parameters():
                param.requires_grad = False
                
        in_features = self.model.classifier[1].in_features
        self.model.classifier[1] = nn.Linear(in_features, num_classes)
        
    def forward(self, x):
        return self.model(x)
