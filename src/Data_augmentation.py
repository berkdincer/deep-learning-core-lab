import torch
from torchvision.transforms import v2
from torchvision.utils import save_image
from torch.utils.data import DataLoader
from CostumDataset import CatsAndDogsDataset

my_transform = v2.Compose([
    v2.RandomHorizontalFlip(p=0.5),
    v2.RandomRotation(degrees=30),
    v2.RandomCrop(size=(224, 224)),
    v2.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4, hue=0.1), # Adjust to the photo.
    v2.RandomGrayscale(p=0.2),
    v2.ToDtype(torch.float32, scale=True),
    v2.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

dataset = CatsAndDogsDataset(root_dir = "for_test/train/train",csv_file="data_aug.csv", transform = my_transform)

img_num = 0
for _ in range(10):
    for image,label in dataset:
        print(image.shape)
        save_image(image,'img'+str(img_num)+'.png')
        img_num += 1


