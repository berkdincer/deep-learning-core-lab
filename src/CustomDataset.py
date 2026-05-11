import os
import pandas as pd
import torch
from torch.utils.data import Dataset
from torchvision.io import read_image

class CatsAndDogsDataset(Dataset):
    def __init__(self,root_dir,csv_file,transform=None):
        self.csv_file = pd.read_csv(csv_file).dropna()
        self.transform = transform
        self.root_dir = root_dir
    def __len__(self):
        return len(self.csv_file)
    def __getitem__(self, index): #Dataloader uses this part.
        img_path = os.path.join(self.root_dir,self.csv_file.iloc[index,0])
        img = read_image(img_path)
        label = torch.tensor(int(self.csv_file.iloc[index,1]))
        if self.transform:
            img = self.transform(img)
        return (img,label)
