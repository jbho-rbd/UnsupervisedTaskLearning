"""======================================================================================
 network.py
 
 Input:   state data labeled with the primitive used at each time step
 Output:  transition model that determines what is the best next primitive to use 
            given the current state variables
 
 Jonathan Ho and Elena Galbally, Fall 2019
======================================================================================"""

""" --------------------------------------------------------------------------------------
   Include Required Libraries and Files
-----------------------------------------------------------------------------------------"""
from dataset_pytorch import PrimitiveTransitionsSet
import torch
import torch.nn as nn            # containing various building blocks for your neural networks
import torch.optim as optim      # implementing various optimization algorithms
import torch.nn.functional as F  # a lower level (compared to torch.nn) interface
from torch.utils.data import Dataset, DataLoader
from time import time
import numpy as np

""" --------------------------------------------------------------------------------------
   Training, Test Sets and Pytorch environment
-----------------------------------------------------------------------------------------"""
trainSet = PrimitiveTransitionsSet('sampleTrain.txt')
trainSet_loader = DataLoader(trainSet, batch_size = 200, shuffle = True, num_workers = 1)
testSet = PrimitiveTransitionsSet('sampleTest.txt')
testSet_loader = DataLoader(testSet, batch_size = 100, shuffle = False, num_workers = 1)

""" --------------------------------------------------------------------------------------
   nn.Module child class: Initializer and Methods
-----------------------------------------------------------------------------------------"""
class Net(nn.Module):
    def __init__(self): # Logistic Regression
        super(Net, self).__init__()
        self.linear = torch.nn.Linear(19, 5) #input, output sizes
    def forward(self, x):
        # print(x.shape)
        x = self.linear(x)
        x = torch.sigmoid(x)
        return x

""" --------------------------------------------------------------------------------------
   GPU, Network Instance and Hyperparameters
-----------------------------------------------------------------------------------------"""
device = torch.device("cuda" if torch.cuda.is_available() else "cpu") #check for GPU
model = Net().to(device)
lossCriterion = torch.nn.BCELoss(size_average=True)
# optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
optimizer = optim.Adam(model.parameters(), lr=0.01, betas=(0.9, 0.999), eps=1e-08, weight_decay=0, amsgrad=False)
# optimizer = torch.optim.SGD(model.parameters(), lr=0.01)

""" --------------------------------------------------------------------------------------
   Network Related Utility Functions
   -----------------------
   * save_checkpoint
   * load_checkpoint
   * test
   * train
------------------------------------------------------------------------------------------- """
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
        for data, target in testSet_loader:   
            data, target = data.to(device), target.to(device)
            output = model(data)
            test_loss += lossCriterion(output, target).item() # sum up batch loss

    test_loss /= len(testSet_loader.dataset)
    print('\nTest set: Average loss: {:.4f} \n'.format(
        test_loss))
    return test_loss

def train(epoch, save_interval = 10, log_interval=1):
    model.train()  # set training mode
    iteration = 0
    traindat = np.zeros((epoch, 3)) 
    for ep in range(epoch):
        start = time()
        for batch_idx, (data, target) in enumerate(trainSet_loader):
            # bring data to the computing device, e.g. GPU
            data, target = data.to(device), target.to(device)

            # forward pass
            output = model(data)
            # compute loss: negative log-likelihood
            loss = lossCriterion(output, target)
            # backward pass
            # clear the gradients of all tensors being optimized.
            optimizer.zero_grad()
            # accumulate (i.e. add) the gradients from this forward pass
            loss.backward()
            # performs a single optimization step (parameter update)
            optimizer.step()
            
            if iteration % log_interval == 0:
                print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                    ep, batch_idx * len(data), len(trainSet_loader.dataset),
                    100. * batch_idx / len(trainSet_loader), loss.item()))
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

""" --------------------------------------------------------------------------------------
   Main
-----------------------------------------------------------------------------------------"""
if __name__ == "__main__":   
    # load_checkpoint('ballnet-1070.pth', model, optimizer)
    train(2)