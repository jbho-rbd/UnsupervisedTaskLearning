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
   Hyperparameters
-----------------------------------------------------------------------------------------"""
BATCH_SIZE = 32
NUM_EPOCHS = 20
LEARNING_RATE = 0.01
BETA_1 = 0.9
BETA_2 = 0.999
EPSILON = 1e-8
INPUT_DIM = 19
OUTPUT_DIM = 6 #num of primitives

SAVE_INTERVAL = 10
PRINT_INTERVAL = 1

""" --------------------------------------------------------------------------------------
   Training, Test Sets and Pytorch environment
-----------------------------------------------------------------------------------------"""
trainSet = PrimitiveTransitionsSet('../data2/run10_labelled')
trainSet_loader = DataLoader(trainSet, batch_size = BATCH_SIZE, shuffle = True, num_workers = 1)
testSet = PrimitiveTransitionsSet('../data2/run10_labelled')
testSet_loader = DataLoader(testSet, batch_size = BATCH_SIZE, shuffle = False, num_workers = 1)

""" --------------------------------------------------------------------------------------
   nn.Module child class: Initializer and Methods
-----------------------------------------------------------------------------------------"""
class Net(nn.Module):
    def __init__(self): # Logistic Regression
        super(Net, self).__init__()
        self.linear = torch.nn.Linear(INPUT_DIM, OUTPUT_DIM) 
    def forward(self, x):
        # print(x.shape)
        x = self.linear(x)
        # x = torch.sigmoid(x)
        return x

""" --------------------------------------------------------------------------------------
   GPU, Network Instance, Optimizer, Loss
-----------------------------------------------------------------------------------------"""
device = torch.device("cuda" if torch.cuda.is_available() else "cpu") #check for GPU
model = Net().to(device)
# lossCriterion = torch.nn.BCELoss(size_average=True)
lossCriterion = nn.CrossEntropyLoss()  
# optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE, betas=(BETA_1, BETA_2), eps=EPSILON, weight_decay=0, amsgrad=False)
# optimizer = torch.optim.SGD(model.parameters(), lr=0.01)

""" --------------------------------------------------------------------------------------
   Network Related Utility Functions
   ----------------------------------
   * save_checkpoint
   * load_checkpoint
   * test
   * train
------------------------------------------------------------------------------------------- """
def save_checkpoint(checkpoint_path, model, optimizer):
    # state_dict: a Python dictionary object that:
    #   - for a model, maps each layer to its parameter tensor;
    #   - for an optimizer, contains info about the optimizers states and hyperparameters used.
    state = {
        'state_dict': model.state_dict(),
        'optimizer' : optimizer.state_dict()}
    torch.save(state, checkpoint_path)
    # print('model saved to %s' % checkpoint_path)

"""-------------------------------------------------------------------------------------------"""
def load_checkpoint(checkpoint_path, model, optimizer):
    state = torch.load(checkpoint_path)
    model.load_state_dict(state['state_dict'])
    optimizer.load_state_dict(state['optimizer'])
    print('model loaded from %s' % checkpoint_path)

"""-------------------------------------------------------------------------------------------"""
def test():
    model.eval()  # set evaluation mode
    batch_test_loss = 0.0
    total = 0
    correct = 0
    count = 0
    with torch.no_grad():
        for state, label in testSet_loader:
            # Load state to a Torch Variable   
            state, label = state.to(device), label.to(device)
            # Forward pass only to get logits/output
            output = model(state)
            # Get predictions from the maximum value
            # Note that, predicted.shape = batch_size
            _, predicted = torch.max(output.data, 1)
            np_predicted = predicted.cpu().detach().numpy()
            if count == 0:
                allPredictions = np_predicted
            else:
                allPredictions = np.hstack((allPredictions,np_predicted))
            count += 1
            # Total number of labels
            total += label.size(0)
            # Total correct predictions
            correct += (predicted == label).sum()
            # Test loss
            test_loss = lossCriterion(output, label)
            # Total batch loss
            batch_test_loss += test_loss.item() 
        
    # Average batch loss
    avg_test_loss = batch_test_loss/len(testSet_loader.dataset)
    # Batch accuracy
    accuracy = 100 * correct / total
    
    # Print accuracy and loss
    print('[--TEST--] Avg Loss: {:.2e}. Accuracy: {:d}'.format(avg_test_loss, accuracy))

    # ----- Save ouput: prediction        
    np.savetxt('test_set_labels/predicted_labels.txt', allPredictions, "%i")
    
    return avg_test_loss, accuracy

"""-------------------------------------------------------------------------------------------"""
def train(num_epochs, save_interval = SAVE_INTERVAL, print_interval=PRINT_INTERVAL):
    model.train()  # set training mode
    iteration = 0
    traindat = np.zeros((num_epochs, 4)) 
    batch_train_loss = 0.0

    for ep in range(num_epochs):        
        start = time()     
        for batch_idx, (data, label) in enumerate(trainSet_loader):
           
            # bring data to the computing device, e.g. GPU
            data, label = data.to(device), label.to(device)
            # forward pass
            output = model(data)
            # compute loss: negative log-likelihood
            train_loss = lossCriterion(output, label)
            # total epoch training loss
            batch_train_loss += train_loss.item() 
            # backward pass
            # clear the gradients of all tensors being optimized.
            optimizer.zero_grad()
            # accumulate (i.e. add) the gradients from this forward pass
            train_loss.backward()
            # performs a single optimization step (parameter update)
            optimizer.step()

            # ----- Save checkpoint (binary file): iteration, model, optimizer
            if iteration % save_interval == 0 and iteration > 0:
                save_checkpoint('checkpoints/transitionModel-%i.pth' % iteration, model, optimizer) 

            # ----- Print epoch progress
            if iteration % print_interval == 0:
                print('Epoch: {} [{}/{} ({:.0f}%)]'.format(ep, 
                    batch_idx * BATCH_SIZE, len(trainSet_loader.dataset),
                    100. * batch_idx / len(trainSet_loader)))

            iteration += 1
        
        """---------------------
            @end of each epoch
        ------------------------"""
        # Print epoch duration
        end = time()
        print('{:.2f}s'.format(end-start)) 
        # Calculate and print average loss
        avg_train_loss = batch_train_loss/len(trainSet_loader.dataset)
        print('[--TRAIN--] Avg Loss: {:.2e}'.format(avg_train_loss))    
        # Evaluate model on testSet and save epoch data
        #    - epoch, test_accuracy, train_loss, test_loss
        traindat[ep,0] = ep # current epoch number
        traindat[ep,2] = avg_train_loss
        avg_test_loss, accuracy = test() # evaluate model on test set
        traindat[ep,1] = accuracy
        traindat[ep,3] = avg_test_loss 
        np.savetxt("model_loss_and_accuracy/accuracy_loss.txt", 
            traindat, ("%i", "%.d", "%.2e", "%.2e"), header='epoch test_accuracy train_avg_loss test_avg_loss')
    
    # Save final checkpoint   
    save_checkpoint('checkpoints/transitionModel-%i.pth' % iteration, model, optimizer)

""" --------------------------------------------------------------------------------------
   Main
-----------------------------------------------------------------------------------------"""
if __name__ == "__main__":   
    # load_checkpoint('ballnet-1070.pth', model, optimizer)
    train(NUM_EPOCHS)
