import torchvision
import torchvision.datasets as dset
import torchvision.transforms as transforms
from torch.utils.data import DataLoader,Dataset
import torchvision.utils
import numpy as np
import random
from PIL import Image
import torch
from torch.autograd import Variable
import PIL.ImageOps    
import torch.nn as nn
from torch import optim
import torch.nn.functional as F

class SiameseNetwork(nn.Module):
    def __init__(self):
        super(SiameseNetwork, self).__init__()
        self.cnn1 = nn.Sequential(
            nn.ReflectionPad2d(1),
            nn.Conv2d(1, 4, kernel_size=3),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(4),
            
            nn.ReflectionPad2d(1),
            nn.Conv2d(4, 8, kernel_size=3),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(8),


            nn.ReflectionPad2d(1),
            nn.Conv2d(8, 8, kernel_size=3),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(8),


        )

        self.fc1 = nn.Sequential(
            nn.Linear(8*227*227, 500),
            nn.ReLU(inplace=True),

            nn.Linear(500, 500),
            nn.ReLU(inplace=True),

            nn.Linear(500, 5))

    def forward_once(self, x):
        output = self.cnn1(x)
        output = output.view(output.size()[0], -1)
        output = self.fc1(output)
        return output
    
    def forward(self, input1, input2):
        output1 = self.forward_once(input1)
        output2 = self.forward_once(input2)
        return output1, output2

class CrossEntropyLoss(torch.nn.Module):
    def __init__(self):
        super(CrossEntropyLoss, self).__init__()

    def forward(self, output, label):
        label = label.long()
        loss = F.cross_entropy(output, label)
        return loss

use_cuda = torch.cuda.is_available()
device = torch.device("cuda" if use_cuda else "cpu")
net = SiameseNetwork().to(device)
criterion = CrossEntropyLoss().to(device)
optimizer = optim.Adam(net.parameters(), lr = 0.001)
counter = []
loss_history = [] 
iteration_number= 0

def train(model, device, train_loader, epoch):
    model.train()
    losses = []
    accurate_labels = 0
    all_labels = 0
    len_train_loader = len(train_loader)
    for batch_idx, (data0, data1, label) in enumerate(tqdm.tqdm(train_loader)): #tqdm.tqdm(train_loader)
        data0, data1, label = data0.to(device), data1.to(device), label.to(device)
        optimizer.zero_grad()
        
        out = model(data0, data1)
        loss_function = criterion(out, label)
        losses.append(loss_function.item())
        loss_function.backward()
        
        optimizer.step()
        
        accurate_labels += torch.sum(torch.argmax(out, dim=1) == label).cpu()
        all_labels += len(label)
            
        if batch_idx % 20 == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}\tTrain Accuracy: {:.6f}'.format(
                epoch, (batch_idx+1) * len(data0), len(train_loader.dataset),
                100. * (batch_idx+1) / len(train_loader), loss_function.item(),
                (100. * accurate_labels / all_labels)))
    train_loss = np.mean(losses)
    train_accuracy = 100. * accurate_labels / all_labels
    print('\nTrain set: Average loss = {:.4f}, Train Accuracy = {:.4f}\n'.format(train_loss, train_accuracy))
    return train_loss, train_accuracy
     


def test(model, device, test_loader):
    model.eval()
    accurate_labels = 0
    all_labels = 0
    losses = []
    with torch.no_grad():
        for batch_idx, (data0, data1, label) in enumerate(tqdm.tqdm(test_loader)):
            data0, data1, label = data0.to(device), data1.to(device), label.to(device)
            out = model(data0, data1)
            loss_function = criterion(out, label)
            losses.append(loss_function.item())

            accurate_labels += torch.sum(torch.argmax(out, dim=1) == label).cpu()
            all_labels += len(label)
    test_loss = np.mean(losses)
    test_accuracy = 100. * accurate_labels / all_labels
    print('\nTest set: Average loss = {:.4f}, Test Accuracy = {:.4f}\n'.format(test_loss, test_accuracy))
    return test_loss, test_accuracy

EPOCHS = 5
