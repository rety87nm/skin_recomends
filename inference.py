#!/usr/bin/env python3

# пример инференса для применения Класса SkinTypeChecker

from SkinTypeChecker import SkinTypeChecker
import matplotlib.pyplot as plt
import numpy as np

model_path = "mobilenetv2.pth"

image_path_acne = "skin_types/test1/acne/19_jpg.rf.231a085ac891ea2bed663ee140b9aa21.jpg"
image_path_dry = "skin_types/test1/dry/dry16_jpg.rf.ee2726f8440e10c24acba66999904b66.jpg"
image_path_oil = "skin_types/test1/oil/images28_jpeg.rf.23fb36cda5a74bb378d89feb6925c5ac.jpg"

sc = SkinTypeChecker(model_path)

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


for path in (image_path_acne, image_path_dry,  image_path_oil):
     label, dlass_id, probs = sc.analyze(path)
     print(label, probs)
     imshow(sc.image())
     plt.show()

# вызов анализа конкретного файла
def analyze(image_path: str) -> str:
    label, class_id, probs = sc.analyze(image_path)
    print(label, probs)
    return label
