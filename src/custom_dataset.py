import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader
import torchvision.datasets as datasets
import torchvision.transforms as transforms
import torchvision
from CustomDataset import (CatsAndDogsDataset)
import os

print(os.getcwd())

#Set device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

#Hyperparameters:
in_channel = 3
batch_size = 1024
epochs = 10
learning_rate = 0.001
num_classes = 10

#Pretrain model:
model = torchvision.models.googlenet(pretrained=True)
model.to(device)
#
dataset = CatsAndDogsDataset(root_dir= "./cat_vs_dog/train/train", csv_file = "train.csv", transform = transforms.ToTensor())
train_data, test_data = torch.utils.data.random_split(dataset, [int(len(dataset)*0.8),int(len(dataset)*0.2)])
train_loader = DataLoader(dataset=train_data, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(dataset=test_data, batch_size=batch_size, shuffle=True)

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(filter(lambda x: x.requires_grad, model.parameters()), lr=learning_rate) # use as parameter which grad values are True


for epoch in range(epochs):
    losses = []
    for batch_idx,(data,target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        scores = model(data)
        loss = criterion(scores, target)
        losses.append(loss.item())
        loss.backward()
        optimizer.step()

def CheckAccuracy(model, test_loader):
    if test_loader.dataset.train:
        print('Train dataset accuracy test.')
    else:
        print('Train dataset accuracy test.')
    correct = 0
    total = 0
    accuracy = 0
    model.eval()
    with torch.no_grad():
        for data,target in test_loader:
            data, target = data.to(device), target.to(device)
            score = model(data)
            _,indices = score.max(dim=1) # we have to check both indices and target because if highest score on the right indices that mean correct guess.
            total += score.size(0)
            correct += (indices == target).sum().item()
            acc = correct / total
    print(f'Got {correct} / {total} with accuracy {float(correct)/float(total)*100:.2f}%')
    model.train() # for training we have to write that if we don't, model still going to be on evaluation mode.
    return acc

CheckAccuracy(model, train_loader)
CheckAccuracy(model, test_loader)