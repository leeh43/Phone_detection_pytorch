import os
import numpy as np
from torch.utils import data
import matplotlib.image
import cv2

class pytorch_loader(data.Dataset):
    def __init__(self, subdict):
        self.subdict = subdict
        self.img_files = subdict['image_file']
        if 'image_label' in subdict:
            self.label_files = subdict['image_label']
            self.num_labels = len(self.label_files)
        else:
            raise ValueError('No labels for training!')

    def __getitem__(self, item):

        num_labels = self.num_labels
        img_file = self.img_files[item]
        img = cv2.imread(img_file)
        img = (img - img.min()) / (img.max() - img.min())
        img = np.transpose(img, (2,0,1))
        x = np.zeros((3, img.shape[1], img.shape[2]))
        x[:, :, :] = img
        x = x.astype('float32')

        label_file = self.label_files[item]
        y = np.array(label_file)
        y = y.astype('float32')

        return x , y

    def __len__(self):
        return len(self.img_files)