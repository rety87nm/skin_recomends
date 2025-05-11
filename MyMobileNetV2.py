# Класс с Моделью на основе MobilenetV2:
import torch
import torch.nn as nn
from torchvision.models import mobilenet_v2, MobileNet_V2_Weights

class MobileNetCnn(nn.Module):
    def __init__(self, num_classes=3, pretrained=True):
        super().__init__()
        
        # Загружаем предобученную MobileNetV2
        weights = MobileNet_V2_Weights.DEFAULT if pretrained else None
        self.model = mobilenet_v2(weights=weights)
        
        # MobileNetV2 возвращает 1280 признаков после последнего слоя, меняем его на свой классификатор:
        self.model.classifier = nn.Sequential(
            nn.Dropout(p=0.2),  
            nn.Flatten(),
            nn.Linear(in_features=1280, out_features=2560),
            nn.Dropout(p=0.3),
            nn.LeakyReLU(),
            nn.Linear(2560, num_classes)
        )

    def forward(self, x):
        logits = self.model(x)
        return logits
