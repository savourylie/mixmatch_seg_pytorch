from glob import glob
import os

import cv2
from PIL import Image
from torchvision import datasets

class CIFAR10(object):

    def __init__(self, root, resize=None, transform=None):
        self.filepath_list = glob(root + '*')
        self.label_dict = {
                            'airplane': 0,
                            'automobile': 1,
                            'bird': 2,
                            'cat': 3,
                            'deer': 4,
                            'dog': 5,
                            'frog': 6,
                            'horse': 7,
                            'ship': 8,
                            'truck': 9
        }

        self.resize = resize
        self.transform = transform
        self.len = len(self.filepath_list)

    def __getitem__(self, index):
        file_path = self.filepath_list[index]
        filename = os.path.basename(file_path)

        file_stem, file_ext = filename.split('.')

        img = cv2.imread(file_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        if self.resize is not None:
            img = cv2.resize(img, (self.resize, self.resize), interpolation=cv2.INTER_CUBIC)

        img_pil = Image.fromarray(img)

        if self.transform is not None:
            img = self.transform(img_pil)

        label = self.label_dict[file_stem.split('_')[-1]]

        return img, label

    def __len__(self):
        return self.len