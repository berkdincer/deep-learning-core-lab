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

#Setting Device:
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

#Hyperparameters:
batch_size = 64
learning_rate = 0.001
num_epochs = 1
input_size = 784
num_classes = 10

#Load Data:
train_dataset = datasets.MNIST(root='./dataset',train = True, transform= transforms.ToTensor(), download=True)
train_loader = DataLoader(dataset = train_dataset, shuffle = True, batch_size = batch_size)
test_dataset = datasets.MNIST(root='./dataset',train = False, transform= transforms.ToTensor(), download=True)
test_loader = DataLoader(dataset = test_dataset, shuffle = False, batch_size = batch_size) # Shuffle must be False for test dataset, it consume unnecessary GPU if it is True.

#Initialize network
model = NN(input_size = input_size, num_classes = num_classes).to(device) # To use defined device on the top .to is needed.

#Loss and optimizer
criterion = nn.CrossEntropyLoss() # Best option for classification.
"""For the regression nn.MSELoss will be the best (Mean Square Error)"""
optimizer = optim.Adam(model.parameters(), lr = learning_rate) # Best one for CV projects. SGD is outdated.

#Train Network
for epoch in range(num_epochs):
    for target_id,(data, target) in enumerate(train_loader):
        target, data = target.to(device), data.to(device)
        #we have to make flatter for nn.Linear layer.
        data = data.reshape(data.shape[0] , -1) # 64,784
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
            data = data.reshape(data.shape[0] , -1)
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