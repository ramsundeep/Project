import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset, Dataset
from torchvision import datasets, transforms, models # add models to the list
from torchvision.utils import make_grid
from sklearn.model_selection import train_test_split
from PIL import Image

import numpy as np
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')


import warnings
warnings.filterwarnings("ignore")


class ConvolutionalNetwork(nn.Module):
    def _init_(self):
        super()._init_()
        self.conv1 = nn.Conv2d(1,10, 3, 1)
        self.conv2 = nn.Conv2d(10, 20, 3, 1)
        self.conv3 = nn.Conv2d(20,40, 3, 1)
        self.fc1 = nn.Linear(17*17*40, 100)
        self.fc2 = nn.Linear(100, 60)
        self.fc3 = nn.Linear(60,25)

    def forward(self, X):
        X = F.relu(self.conv1(X))
        X = F.max_pool2d(X, 2, 2)
        X = F.relu(self.conv2(X))
        X = F.max_pool2d(X, 2, 2)
        X = F.relu(self.conv3(X))
        X = F.max_pool2d(X, 2, 2)
        X = X.view(-1, 17*17*40)
        X = F.relu(self.fc1(X))
        X = F.relu(self.fc2(X))
        X = self.fc3(X)
        return F.log_softmax(X, dim=1)

mdl="model.pth"    
model=ConvolutionalNetwork()
model.load_state_dict(torch.load(mdl))
model.eval()


def test_func(X):
    X=X.reshape(X.shape[0],1,150,150)
    X=X/255
    X_testTensor = torch.Tensor(X)
    y_pred=model(X_testTensor)
    predicted = torch.max(y_pred.data, 1)[1]
    predicted=predicted.numpy()
    return predicted


X = np.load("Images.npy")
y_train=np.load('Labels.npy')
predicted=test_func(X)
correct=(predicted == y_train).sum()
accuracy=correct/y_train.shape*100
print(accuracy)