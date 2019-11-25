from Q4_helper import load_dataset, load_training_dataset, load_testing_dataset
from dataset_pytorch import BallTrackSet
import torch
import torch.nn as nn            # containing various building blocks for your neural networks
import torch.optim as optim      # implementing various optimization algorithms
import torch.nn.functional as F  # a lower level (compared to torch.nn) interface
from torch.utils.data import Dataset, DataLoader
from resnet import resnet34, resnet18
from time import time
from spatialsoftmax import SpatialSoftmax
import numpy as np


images,labels = load_training_dataset('data/Q4A_data/training_set/Q4A_positions_train.npy',
                                        'data/Q4A_data/training_set')
Q4A_train_set = BallTrackSet(images,labels)
Q4A_train_set_loader = DataLoader(Q4A_train_set, batch_size = 200, shuffle = True, num_workers = 1)
images,labels = load_testing_dataset('data/Q4A_data/testing_set/Q4A_positions_test.npy',
                                        'data/Q4A_data/testing_set')
Q4A_test_set = BallTrackSet(images,labels)
Q4A_test_set_loader = DataLoader(Q4A_test_set, batch_size = 100, shuffle = False, num_workers = 1)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        
        # Conv2d(in_channels, out_channels, kernel_size, stride=1, padding=0,
        #        dilation=1, groups=1, bias=True, padding_mode='zeros')
        self.conv1 = nn.Conv2d(3, 16, kernel_size=7, stride=2, padding=3)
        self.bn1 = nn.BatchNorm2d(16)
        self.max_pool = nn.MaxPool2d(3)
        self.conv2 = nn.Conv2d(16, 16, kernel_size=3, stride=1)
        self.spatialsoftmax1 = SpatialSoftmax(80, 106, 16, temperature = 1)
        
        # Linear(in_features, out_features, bias=True)
        
        # MaxPool2d(kernel_size, stride=None, padding=0, dilation=1, return_indices=False, ceil_mode=False)
        # ReLU(inplace=False)
        self.relu = nn.ReLU(inplace=True)
        self.fc = nn.Linear(32, 3)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.max_pool(x)
        # for i in range(3):
        #     x = self.conv2(x)
        #     x = self.bn1(x)
        # print(x.shape)
        x = self.spatialsoftmax1(x)
        # print(x.shape)
        # print(x.shape)
        x = self.fc(x)
        return x

model = Net().to(device)
# model = resnet18(num_classes=3).to(device)
#learning rate zero
# optimizer = optim.SGD(model.parameters(), lr=0.000, momentum=0.9)
optimizer = optim.Adam(model.parameters(), lr=0.01, betas=(0.9, 0.999), eps=1e-08, weight_decay=0, amsgrad=False)
def save_checkpoint(checkpoint_path, model, optimizer):
    # state_dict: a Python dictionary object that:
    # - for a model, maps each layer to its parameter tensor;
    # - for an optimizer, contains info about the optimizers states and hyperparameters used.
    state = {
        'state_dict': model.state_dict(),
        'optimizer' : optimizer.state_dict()}
    torch.save(state, checkpoint_path)
    print('model saved to %s' % checkpoint_path)
    
def load_checkpoint(checkpoint_path, model, optimizer):
    state = torch.load(checkpoint_path)
    model.load_state_dict(state['state_dict'])
    optimizer.load_state_dict(state['optimizer'])
    print('model loaded from %s' % checkpoint_path)

def test():
    model.eval()  # set evaluation mode
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in Q4A_test_set_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            test_loss += F.mse_loss(output, target, size_average=False).item() # sum up batch loss

    test_loss /= len(Q4A_test_set_loader.dataset)
    print('\nTest set: Average loss: {:.4f} \n'.format(
        test_loss))
    return test_loss
def train(epoch, save_interval = 10, log_interval=1):
    model.train()  # set training mode
    iteration = 0
    traindat = np.zeros((epoch, 3)) 
    for ep in range(epoch):
        start = time()
        for batch_idx, (data, target) in enumerate(Q4A_train_set_loader):
            # bring data to the computing device, e.g. GPU
            data, target = data.to(device), target.to(device)

            # forward pass
            output = model(data)
            # compute loss: negative log-likelihood
            loss = F.mse_loss(output, target)
            # backward pass
            # clear the gradients of all tensors being optimized.
            optimizer.zero_grad()
            # accumulate (i.e. add) the gradients from this forward pass
            loss.backward()
            # performs a single optimization step (parameter update)
            optimizer.step()
            
            if iteration % log_interval == 0:
                print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                    ep, batch_idx * len(data), len(Q4A_train_set_loader.dataset),
                    100. * batch_idx / len(Q4A_train_set_loader), loss.item()))
            if iteration % save_interval == 0 and iteration > 0:
                save_checkpoint('ballnet-%i.pth' % iteration, model, optimizer)
            iteration += 1
            
        end = time()
        print('{:.2f}s'.format(end-start))
        traindat[ep,0] = ep
        traindat[ep,1] = loss.item()
        traindat[ep,2] = test() # evaluate at the end of epoch
        np.savetxt("train4.dat", traindat)
    save_checkpoint('ballnet-%i.pth' % iteration, model, optimizer)

if __name__ == "__main__":
    load_checkpoint('ballnet-1070.pth', model, optimizer)
    train(500)