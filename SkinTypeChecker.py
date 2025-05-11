# Класс для работы с распознаванием типа кожи по фото:

import torch
import torch.nn as nn
from MyMobileNetV2 import MobileNetCnn
from SkinDataset import SkinDataset
from pathlib import Path
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
import numpy as np

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class SkinTypeChecker():
    def __init__(self, model_path=""):
        super().__init__()
        self.classes = ['acne', 'dry', 'oil', 'normal', 'combine']
        self.model = MobileNetCnn(pretrained=False).to(device)
        self.model.load_state_dict( torch.load(model_path, map_location=device) )

    def analyze(self, image_path: str):

        image_path = Path(image_path)

        self.model.eval()

        self.dataset = SkinDataset([image_path], augmentation=False)

        loader = DataLoader(self.dataset, batch_size=len(self.dataset), shuffle=False)
        data, _ = next(iter(loader))
        data = data.to(device)

        logits = self.model(data)
        predicts = self.ext_labels(logits)

        class_id = predicts[0].item()

        # заклинание на вытаскивание вероятностей распределения по классам в numpy формате:
        probs = logits[0].softmax(dim=-1).round(decimals=2).cpu().detach().numpy()

        return (self.classes[class_id], class_id, probs)

    def image(self):
        if (len(self.dataset)):
            img, _ = self.dataset[0]
            return img

        return None
        
    def ext_labels(self, logits):
        """
        logits: torch.Tensor формы [batch_size, 3] с логитами для классов [acne, dry, oil]
        return: torch.Tensor формы [batch_size] с индексами классов (0=acne, 1=dry, 2=oil, 3=normal, 4=combine)
        """
        
        probs = torch.softmax(logits, dim=-1)  # [batch_size, 3]
        acne_prob, dry_prob, oil_prob = probs[:, 0], probs[:, 1], probs[:, 2] 

        # Условие для normal: max(probs) - min(probs) < 0.5
        max_diff = probs.max(dim=-1)[0] - probs.min(dim=-1)[0]
        is_normal = max_diff < 0.5

        # Условие для combine: dry_prob > 0.3 и oil_prob > 0.3 и (dry_prob + oil_prob) > 0.7
        is_combine = (dry_prob > 0.2) & (oil_prob > 0.2) & ((dry_prob + oil_prob) > 0.7)

        default_labels = probs.argmax(dim=-1)

        # Формируем результат: выбираем классы на основе условий
        labels = default_labels.clone() 
        labels[is_combine] = 4  # combine
        labels[is_normal] = 3   # normal

        return labels
