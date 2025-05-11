#!/usr/bin/env python3

# Классификация изображений из датесета 
# Датасет взять тут: https://github.com/MdAliAhnaf/Skin_Type_Classification-Recommendation/tree/main/skin_dataset

import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import TensorDataset, DataLoader, Dataset
from sklearn.model_selection import train_test_split

import shutil
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from pathlib import Path
from PIL import Image

from tqdm import tqdm
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import LabelEncoder
from torchvision import models

# Свои классы:
from SkinDataset import SkinDataset
from MyMobileNetV2 import MobileNetCnn

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"device: {device}")

data_path = "/home/alexx/deep_learning/data/skin_types"

def imshow(inp, title=None, plt_ax=plt, default=False):
    """Imshow для тензоров"""
    inp = inp.numpy().transpose((1, 2, 0))
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    inp = std * inp + mean
    inp = np.clip(inp, 0, 1)
    plt_ax.imshow(inp)
    if title is not None:
        plt_ax.set_title(title)
    plt_ax.grid(False)


# уникализируем все изображения:
source_data_path = Path( "/home/alexx/deep_learning/data/skin_types/train/" )
clear_data_path = "/home/alexx/deep_learning/data/uniq/skin_types/train/"

files = sorted( list(source_data_path.rglob('*.jpg')) )

d = {}
need_clear = 0
if (need_clear):
    for file in files:
        f = str(file)
        orig_name = f[0:f.find(".")]

        if orig_name in d.keys():
            continue

        d[orig_name] = f

        to_file = Path( clear_data_path + orig_name[orig_name.find("train"):] + ".jpg" )
        
        to_file.parent.mkdir(parents=True, exist_ok=True)

        print (f"{file} --->  {to_file}")
        shutil.copy(file, to_file) 

# Берем очищенные данные как тренировочные и разделяем фалы на датасеты:
TRAIN_DIR = Path( clear_data_path )

all_files = sorted( list(TRAIN_DIR.rglob('*.jpg')) )

train_files, test_files = train_test_split(all_files, test_size=0.2, random_state=42)

train_dataset = SkinDataset(train_files, augmentation=True )
val_dataset = SkinDataset(test_files, augmentation=False )

print( len(train_dataset), len(val_dataset) )

# отобразим данные:
fig, ax = plt.subplots(nrows=3,
                       ncols=3,
                       figsize=(8, 8),
                       sharey=True,
                       sharex=True)

for fig_x in ax.flatten():
    img_id = np.random.randint(0, len(val_dataset) - 9)
    im_val, label = val_dataset[img_id]
    img_label = val_dataset.label_encoder.inverse_transform([label])[0]
    imshow(im_val.data.cpu(),title=img_label, plt_ax=fig_x) 

#plt.show()
#exit()

# пробуем чистую сверточную сеть:
class SimpleCnn(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=8, kernel_size=3),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2)
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(in_channels=8, out_channels=16, kernel_size=3),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2)
        )
        self.conv3 = nn.Sequential(
            nn.Conv2d(in_channels=16, out_channels=32, kernel_size=3),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2)
        )
        self.conv4 = nn.Sequential(
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2)
        )
        self.conv5 = nn.Sequential(
            nn.Conv2d(in_channels=64, out_channels=96, kernel_size=3),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2)
        )

        self.out = nn.Linear(96 * 5 * 5, 3)

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        x = self.conv5(x)

        x = x.view(x.size(0), -1)
        logits = self.out(x)
        return logits

batch_size = 32
epochs = 100
model = SimpleCnn().to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.0001)

train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4)
valid_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=4 )

out_metrix = {'acc':[], 'vloss':[], 'tloss':[]}

# Функция проверки качества модели на сетовой выборке
def validate(model, valid_loader):
    model.eval()
    losess = []

    y_pred, y_true, y_scores = [], [], []

    with torch.no_grad():
        for batch_idx, (data, target) in tqdm(enumerate(valid_loader)):
            data, target = data.to(device), target.to(device)
            outputs = model(data)

            loss = criterion( outputs, target.long() )

            losess.append( loss.item() )

            predicts = torch.argmax(outputs, 1)

            y_pred.extend( predicts.cpu().detach().numpy() )
            y_true.extend( target.cpu().detach().numpy() )

        acc = accuracy_score(y_true, y_pred)

    return np.mean(losess), acc

# отобразить графики последнего обучения
def show_metrix(out_metrix, title=""):
    # строим графики Precision, recall, accuracy:
    fig1, ax1 = plt.subplots()
    ax1.plot(out_metrix['acc'], label='Accuracy' )
    ax1.set_title(f"График точности предсказаний {title}")
    ax1.set_xlabel(f"Эпохи")
    ax1.set_ylabel('Точность (%)')
    ax1.grid(True)
    ax1.legend()

    # отслеживаем значения loss функции
    fig2, ax2 = plt.subplots()
    ax2.plot(out_metrix['tloss'], label='Train Loss')
    ax2.set_title(f'Графики потерь {title}')
    ax2.set_xlabel(f"Эпохи")
    ax2.set_ylabel('Потери')
    ax2.plot(out_metrix['vloss'], label='Valid Loss' )
    ax2.grid(True)
    ax2.legend()
    plt.show()

# функция тренировки модели:
def train_model(model, train_loader, valid_loader, epochs):
    for epoch in range(1, epochs+1):
        print('############# Epoch {}: Training Start   #############'.format(epoch))
        model.train()
        tloss = []

        for i, (data, target) in enumerate(tqdm(train_loader)):
            data, target = data.to(device), target.to(device)
            outputs = model(data)

            loss = criterion(outputs, target.long())
            tloss.append(loss.item())

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        out_metrix['tloss'].append(np.mean(tloss))

        # на каждой эпохе вычисляем метрики
        vl, vacc = validate (model, valid_loader)

        out_metrix['vloss'].append(vl)
        out_metrix['acc'].append(vacc)
        tloss = np.mean(tloss)

        print(f'Epoch: {epoch} Training Loss: {tloss:.4f} Val Loss: {vl:.4f} Accuracy: {vacc:.4f}')


#train_model(model, train_loader, valid_loader, epochs)
#show_metrix(out_metrix, "модель SimpleCnn")

# Усложняем сверточные слои:
class SimpleCnn1(nn.Module):
    def __init__(self, in_channels: int = 3, num_of_classes: int = 3):
        super(SimpleCnn1, self).__init__()

        self.feature_extractor = nn.Sequential(
            # используем padding чтобы не терять информацию по краям
            nn.Conv2d(in_channels=in_channels, out_channels=32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.Dropout2d(p=0.2),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2), # 3x224x224 -> 32x112x112

            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.Dropout2d(p=0.2),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2), # 32x112x112 -> 64x56x56

            nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.Dropout2d(p=0.2),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2), # 64x56x56 -> 128x28x28

            nn.Conv2d(in_channels=128, out_channels=256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.Dropout2d(p=0.2),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2), # 128x28x28 -> 256x14x14

            nn.Conv2d(in_channels=256, out_channels=512, kernel_size=3, padding=1),
            nn.BatchNorm2d(512),
            nn.Dropout2d(p=0.2),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2) # 256x14x14 -> 512x7x7
        )

        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(in_features=512 * 7 * 7, out_features=4096),
            nn.Dropout(p=0.3),
            nn.LeakyReLU(),
            nn.Linear(4096, num_of_classes)
        )

    def forward(self, x):
        x = self.feature_extractor(x)
        probs = self.classifier(x)
        return probs

epochs = 100
model = SimpleCnn1().to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.003)

#train_model(model, train_loader, valid_loader, epochs)
#show_metrix(out_metrix, "модель SimpleCnn")

model = MobileNetCnn(pretrained=False).to(device)
print(model)
epochs = 500
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.003)

train_model(model, train_loader, valid_loader, epochs)
show_metrix(out_metrix, "модель SimpleCnn")

model_path = "/home/alexx/deep_learning/data/skin_types/mobilenetv2.pth"
torch.save( model.state_dict(), model_path )
