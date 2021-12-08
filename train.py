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



images = np.load("Train_Images.npy")
labels = np.load("Train_Labels.npy").T



labels.shape



images = images.reshape(23502,1,150,150)
images.shape



X_train, X_test, y_train, y_test = train_test_split(images, labels, test_size=0.33, random_state=42)



X_train = X_train/255
X_test = X_test/255



X_trainTensor = torch.Tensor(X_train)
X_testTensor = torch.Tensor(X_test)
y_trainTensor = torch.Tensor(y_train)
y_testTensor = torch.Tensor(y_test)



X_trainTensor[0]



y_trainTensor = y_trainTensor.type(torch.LongTensor)
y_testTensor = y_testTensor.type(torch.LongTensor)



train_data = TensorDataset(X_trainTensor,y_trainTensor) # create your datset
train_loader = DataLoader(train_data,batch_size=10,shuffle=True) # create your dataloader


test_data = TensorDataset(X_testTensor,y_testTensor) # create your datset
test_loader = DataLoader(test_data,batch_size=10,shuffle=True) # create your dataloader



len(train_data)



len(test_data)



print(f'Training images available: {len(train_data)}')
print(f'Testing images available:  {len(test_data)}')



class ConvolutionalNetwork(nn.Module):
    def __init__(self):
        super().__init__()
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



(((150-2)/2)-2)/2



torch.manual_seed(101)
CNNmodel = ConvolutionalNetwork()
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(CNNmodel.parameters(), lr=0.001)
CNNmodel



def count_parameters(model):
    params = [p.numel() for p in model.parameters() if p.requires_grad]
    for item in params:
        print(f'{item:>8}')
    print(f'________\n{sum(params):>8}')



count_parameters(CNNmodel)



import time
start_time = time.time()

epochs = 3

max_trn_batch = 3000
max_tst_batch = 3000

train_losses = []
test_losses = []
train_correct = []
test_correct = []

for i in range(epochs):
    trn_corr = 0
    tst_corr = 0
    
    # Run the training batches
    for b, (X_train, y_train) in enumerate(train_loader):
        
        # Limit the number of batches
        if b == max_trn_batch:
            break
        b+=1
        
        # Apply the model
        y_pred = CNNmodel(X_train)
        loss = criterion(y_pred, y_train)
 
        # Tally the number of correct predictions
        predicted = torch.max(y_pred.data, 1)[1] 
        batch_corr = (predicted == y_train).sum()
        trn_corr += batch_corr
        
        # Update parameters
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # Print interim results
        if b%100 == 0:
            print(f'epoch: {i:2}  batch: {b:4} [{10*b:6}/8000]  loss: {loss.item():10.8f} accuracy: {trn_corr.item()*100/(10*b):7.3f}%')

    train_losses.append(loss)
    train_correct.append(trn_corr)

    # Run the testing batches
    with torch.no_grad():
        for b, (X_test, y_test) in enumerate(test_loader):
            # Limit the number of batches
            if b == max_tst_batch:
                break

            # Apply the model
            y_val = CNNmodel(X_test)

            # Tally the number of correct predictions
            predicted = torch.max(y_val.data, 1)[1] 
            tst_corr += (predicted == y_test).sum()

    loss = criterion(y_val, y_test)
    test_losses.append(loss)
    test_correct.append(tst_corr)

print(f'\nDuration: {time.time() - start_time:.0f} seconds') # print the time elapsed



plt.plot([t/157.46 for t in train_correct], label='training accuracy')
plt.plot([t/77.56 for t in test_correct], label='validation accuracy')
plt.title('Accuracy at the end of each epoch')
plt.legend();


Training images available: 15322
Testing images available:  8180



for i in train_correct:
    print(i/15322)



for i in test_correct:
    print(i/8180)



file="model.pth"
torch.save(CNNmodel.state_dict(),file)