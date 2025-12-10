import torch
import torch.nn as nn
from torchvision import models
from torchvision.models import EfficientNet_B4_Weights

class EmotionEfficientNetB4(nn.Module):
    def __init__(self, num_classes=7, dropout_prob=0.5):
        super(EmotionEfficientNetB4, self).__init__()
        weights = EfficientNet_B4_Weights.DEFAULT
        self.model = models.efficientnet_b4(weights=weights)
        in_features = self.model.classifier[1].in_features
        self.model.classifier = nn.Sequential(
            nn.Dropout(dropout_prob),
            nn.Linear(in_features, num_classes)
        )

    def forward(self, x):
        return self.model(x)
