import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader
import torchvision.datasets as datasets
import torchvision.transforms as transforms

#Creating
class NN(nn.Module): ## nn.Module has inhareted.
    def __init__(self,input_size,num_classes):
        super(NN, self).__init__()
        self.fcl1 = nn.Linear(input_size,64)
        self.fcl2 = nn.Linear(64,num_classes)

    def forward(self,x): ## Name of function must be forward, nn.Module has its forward function if forward not defined, it will raise an error message. Thus, forward fuction is a must!
        x = F.relu(self.fcl1(x))
        x = self.fcl2(x)
        return x

class CNN(nn.Module):
    def __init__(self,in_channel = 1, num_classes = 10): # Channel size is must cuz it adjust its depth according to in_channel's number.
        super(CNN, self).__init__()
        self.conv1 = nn.Conv2d(in_channels = in_channel, out_channels = 8, kernel_size = (3,3), stride = (1,1), padding=(1,1)) # Out_channel will be increased if dataset getting complex.
        self.pool = nn.MaxPool2d(kernel_size=(2,2), stride=(2,2))
        self.conv2 = nn.Conv2d(in_channels=8, out_channels=16, kernel_size=(3, 3), stride=(1, 1),padding=(1, 1))
        self.fcl1 = nn.Linear(16*7*7, num_classes)

    def forward(self,x):
        x = F.relu(self.conv1(x))
        x = self.pool(x)
        x = F.relu(self.conv2(x))
        x = self.pool(x)
        x = x.reshape(x.shape[0],-1)
        x = self.fcl1(x)
        return x

#Setting Device:
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

#Hyperparameters:
batch_size = 64
learning_rate = 0.001
num_epochs = 1
in_channel = 1
num_classes = 10

#Load Data:
train_dataset = datasets.MNIST(root='./dataset',train = True, transform= transforms.ToTensor(), download=True)
train_loader = DataLoader(dataset = train_dataset, shuffle = True, batch_size = batch_size)
test_dataset = datasets.MNIST(root='./dataset',train = False, transform= transforms.ToTensor(), download=True)
test_loader = DataLoader(dataset = test_dataset, shuffle = False, batch_size = batch_size) # Shuffle must be False for test dataset, it consume unnecessary GPU if it is True.

#Initialize network
model = CNN().to(device) # To use defined device on the top .to is needed.

#Loss and optimizer
criterion = nn.CrossEntropyLoss() # Best option for classification.
"""For the regression nn.MSELoss will be the best (Mean Square Error)"""
optimizer = optim.Adam(model.parameters(), lr = learning_rate) # Best one for CV projects. SGD is outdated.


#Train Network
for epoch in range(num_epochs):
    for target_id,(data, target) in enumerate(train_loader):
        target, data = target.to(device), data.to(device)
        #we have to make flatter for nn.Linear layer.
        optimizer.zero_grad()
        #forward propagation
        model_last = model(data)
        loss = criterion(model_last, target)
        #backward propagation
        loss.backward()
        #adam step
        optimizer.step()

#Check the accuracy on training & test:

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