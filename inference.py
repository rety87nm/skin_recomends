#!/usr/bin/env python3

# пример инференса для применения Класса SkinTypeChecker

from SkinTypeChecker import SkinTypeChecker
import matplotlib.pyplot as plt
import numpy as np
import config

image_path_acne = "/home/alexx/deep_learning/data/uniq/skin_types/test1/acne/97_jpg.rf.8230f82167835d4948902f3a5dee9232.jpg"

sc = SkinTypeChecker(config.model_path)

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


label, dlass_id, probs = sc.analyze(image_path_acne)
print(label, probs)
imshow(sc.image())
plt.show()

# вызов анализа конкретного файла
def analyze(image_path: str) -> str:
    label, class_id, probs = sc.analyze(image_path)
    print(label, probs)
    return label
