import torch
from torch.utils.data import Dataset
import glob
import numpy as np


class QuanvDataset(Dataset):
    def __init__(self, path):
        self.imgs_path = path
        file_list = glob.glob(self.imgs_path + "*")
        self.data = []
        for i, class_path in enumerate(file_list):
            print(f"{class_path}: {i}")
            for img_path in glob.glob(class_path + "/*.npy"):
                self.data.append([img_path, i])
    def __len__(self):
        return len(self.data)
    def __getitem__(self, idx):
        img_path, class_id = self.data[idx]
        img = np.load(img_path)
        img = np.stack((img[:,:,0], img[:,:,1], img[:,:,2], img[:,:,3]))
        img_tensor = torch.from_numpy(img.astype(np.float32))
        return img_tensor, class_id