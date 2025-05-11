from torch.utils.data import Dataset
from sklearn.preprocessing import LabelEncoder
from PIL import Image
import torchvision
import torchvision.transforms as transforms
import numpy as np

class SkinDataset(Dataset):
    """
    Датасет загружает файлы, и формирует метки 
    """
    def __init__(self, files, augmentation=True, rescale=224):
        super().__init__()
        
        self.files = sorted(files)
        self.augmentation = augmentation
        self.rescale = rescale

        # из названий директорий делаем one-hot метки
        self.label_encoder = LabelEncoder()
        self.labels = [path.parent.name for path in self.files]
        self.label_encoder.fit(self.labels)

    def __len__(self):
        return len(self.files)

    def load_sample(self, file):
        image = Image.open(file)
        image.load()
        return image

    def __getitem__(self, index):
        x = self.load_sample(self.files[index])
        x = self._prepare_sample(x)
        x = Image.fromarray(np.uint8(x))

        if self.augmentation:
            transform = transforms.Compose([
                transforms.RandomHorizontalFlip(p=0.5),
                transforms.RandomVerticalFlip(p=0.5),
                transforms.RandomRotation(degrees=30),

                transforms.ColorJitter(
                    brightness=0.2,
                    contrast=0.2,
                    saturation=0.2,
                    hue=0.1
                ),

                transforms.RandomApply([
                    transforms.GaussianBlur(kernel_size=5, sigma=(0.1, 2.0))
                ], p=0.3),

                # изменение резкозти
                transforms.RandomAdjustSharpness(sharpness_factor=1.5, p=0.3),

                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
            ])
        else:
            transform = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
            ])

        x = transform(x)

        label = self.labels[index]
        label_id = self.label_encoder.transform([label])
        y = label_id.item()
        
        return x, y

    # предварительная обработка перед трансформациями:
    def _prepare_sample(self, image):
        image = image.resize((self.rescale, self.rescale))
        return np.array(image)

