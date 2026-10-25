import torch
import torch.nn as nn
import torchvision.models as models

#pre-trained model as feature extractor
class CatDogModel(nn.Module):
    def __init__(self, num_classes=2, freeze_features=True):
        super().__init__()
        
        #the pre-trained model
        self.model = models.mobilenet_v2(weights=models.MobileNet_V2_Weights.DEFAULT)
        
        #freazing the cnn features(bottelneck)
        if freeze_features:
            for param in self.model.features.parameters():
                param.requires_grad = False
        #in_features=1280        
        in_features = self.model.classifier[1].in_features
        self.model.classifier[1] = nn.Linear(in_features, num_classes)
        
    def forward(self, x):
        return self.model(x)
