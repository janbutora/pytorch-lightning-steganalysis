import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)
import numpy as np
import albumentations as A
from albumentations.pytorch.transforms import ToTensorV2
from torch.utils.data import Dataset
from tools.jpeg_utils import *
from PIL import Image



def get_train_transforms(size=512):
    return A.Compose([
            A.HorizontalFlip(p=0.5),
            A.VerticalFlip(p=0.5),
            A.RandomRotate90(p=1.0),
            A.Resize(height=size, width=size, p=1.0),
            A.Normalize(mean=0, std=1, max_pixel_value=255.),
            ToTensorV2(p=1.0),
        ], p=1.0, additional_targets={'image2':'image'})


def get_valid_transforms(size=512):
    return A.Compose([
            A.Resize(height=size, width=size, p=1.0),
            A.Normalize(mean=0, std=1, max_pixel_value=255.),
            ToTensorV2(p=1.0),
        ], p=1.0, additional_targets={'image2':'image'})


class Retriever(Dataset):

    def __init__(self, image_names, labels,  transforms=None, return_name=False):
        super().__init__()
        self.image_names = image_names
        self.return_name = return_name
        self.labels = labels
        self.transforms = transforms


    def __getitem__(self, index: int):

        image_name, label = self.image_names[index], self.labels[index]

        image = np.array(Image.open(f'{image_name}')).astype(np.uint8)
        if self.transforms:
            sample = {'image': image}
            sample = self.transforms(**sample)
            image = sample['image']

        if self.return_name:
            return image, label, image_name
        else:
            return image, label

    def __len__(self) -> int:
        return self.image_names.shape[0]

    def get_labels(self):
        return list(self.labels)
