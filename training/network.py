"""======================================================================================
 network.py
 
 Input:   state data labeled with the primitive used at each time step
 Output:  transition model that determines what is the best next primitive to use 
            given the current state variables
 
Elena Galbally and Jonathan Ho, Fall 2019
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
import numpy as np
import argparse
import time

""" --------------------------------------------------------------------------------------
   Global Constants
-----------------------------------------------------------------------------------------"""
INPUT_DIM = 19
NUM_CLASSES = 6 #num of primitives
TRAIN_RUNS = 3 #actually 14 because it starts at 1, we are missing 11 and 16 was trash
TOTAL_RUNS = 20
MID_LAYER_DIM = 100

""" --------------------------------------------------------------------------------------
   Command Line Arguments: Network Selection
-----------------------------------------------------------------------------------------"""
parser = argparse.ArgumentParser(description='PyTorch Transistion Model Training')
parser.add_argument('--modelid', required = True, type=int, metavar='MODEL_ID',
                    help='1(linear), 2(superNet)')
args = parser.parse_args()
MODEL_ID = args.modelid 

""" --------------------------------------------------------------------------------------
   Utility Functions
-----------------------------------------------------------------------------------------"""
def create_txt_file_name(folder_path, description_string, model_name):
    timestamp_string = time.strftime("%Y%m%d-%H%M%S") 
    filename = folder_path + model_name + '_' + timestamp_string + '_' + description_string + '.txt'
    return filename    

""" --------------------------------------------------------------------------------------
   Network Architectures (nn.Module child class) 
-----------------------------------------------------------------------------------------"""
class LogisticRegression(nn.Module):
    def __init__(self): # Logistic Regression
        super(LogisticRegression, self).__init__()
        self.linear = torch.nn.Linear(INPUT_DIM, NUM_CLASSES) 
    def forward(self, x):
        x = self.linear(x)
        return x

class SuperNet(nn.Module):
    def __init__(self): 
        super(SuperNet, self).__init__()
        self.linear1 = torch.nn.Linear(INPUT_DIM, MID_LAYER_DIM) 
        self.linear2 = torch.nn.Linear(MID_LAYER_DIM, NUM_CLASSES) 
    def forward(self, x):
        # print(x.shape)
        x = self.linear1(x)
        x = torch.sigmoid(x)
        x = self.linear2(x)
        return x

""" --------------------------------------------------------------------------------------
   Network Training And Testing Class 
-----------------------------------------------------------------------------------------"""
class NeuralNetwork:
    """ 
    Class that initializes the model with the correct hyperparameters and optimizer
    Class functions:
        -- load_datasets
        -- save_checkpoints
        -- load_checkpoints
        -- test
        -- train
    """

    """---------------------
        Initializer
    ------------------------"""
    def __init__(self, MODEL_ID):

        if MODEL_ID == 1:
            self.MODEL_NAME = "logisticRegression"
            self.OPTIM_NAME = "Adam"
            self.LOSS_NAME = "CrossEntropy"
            self.BATCH_SIZE = 32
            self.NUM_EPOCHS = 100
            self.SAVE_INTERVAL = 10
            self.PRINT_INTERVAL = 30
            self.LEARNING_RATE = 0.01
            self.BETA_1 = 0.9
            self.BETA_2 = 0.999
            self.EPSILON = 1e-8
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu") #check for GPU
            self.model = LogisticRegression().to(self.device)
            self.lossCriterion = nn.CrossEntropyLoss()  
            self.optimizer = optim.Adam(self.model.parameters(), 
                lr=self.LEARNING_RATE, betas=(self.BETA_1, self.BETA_2), 
                eps=self.EPSILON, weight_decay=0, amsgrad=False)
    
        elif MODEL_ID == 2:
            self.MODEL_NAME = "superNet"
            self.OPTIM_NAME = "Adam"
            self.LOSS_NAME = "CrossEntropy"
            self.BATCH_SIZE = 32
            self.NUM_EPOCHS = 200
            self.SAVE_INTERVAL = 10
            self.PRINT_INTERVAL = 40
            self.LEARNING_RATE = 0.01
            self.BETA_1 = 0.9
            self.BETA_2 = 0.999
            self.EPSILON = 1e-8
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu") #check for GPU
            self.model = SuperNet().to(self.device)
            self.lossCriterion = nn.CrossEntropyLoss()  
            self.optimizer = optim.Adam(self.model.parameters(), 
                lr=self.LEARNING_RATE, betas=(self.BETA_1, self.BETA_2), 
                eps=self.EPSILON, weight_decay=0, amsgrad=False)
        
        else:
            raise ValueError('Model ID must be an integer between 1 and 2')

    """---------------------
        Pytorch Data sets
    ------------------------"""
    def load_datasets(self):
        # A) For training and testing on multiple runs
        train_data_list = []
        test_data_list = []

        for train_run_number in range(TRAIN_RUNS):
            if (train_run_number == 0):
                print('[INFO] Runs start at #1 not #0')
            elif (train_run_number == 11):
                print('[INFO] We lost the data from run #11')
            elif (train_run_number == 16):
                print('[INFO] run #16 was a piece of shit')
            else:
                newString = '../data/medium_cap/auto_labelled/run{:d}_labelled'.format(train_run_number)
                train_data_list.append(newString)

        # for test_run_number in range(TRAIN_RUNS,TOTAL_RUNS):
        for test_run_number in range(17,TOTAL_RUNS):
            newString = '../data/medium_cap/auto_labelled/run{:d}_labelled'.format(test_run_number)
            test_data_list.append(newString)

        trainSet = PrimitiveTransitionsSet(train_data_list)
        testSet = PrimitiveTransitionsSet(test_data_list)

        # B) For training and testing on single runs
        # trainSet = PrimitiveTransitionsSet('../data/medium_cap/auto_labelled/run10_labelled')
        # testSet = PrimitiveTransitionsSet('../data/medium_cap/auto_labelled/run12_labelled')

        # Pytorch data set
        self.trainSet_loader = DataLoader(trainSet, batch_size = self.BATCH_SIZE, shuffle = True, num_workers = 1)
        self.testSet_loader = DataLoader(testSet, batch_size = self.BATCH_SIZE, shuffle = False, num_workers = 1)
    
    """---------------------
        Checkpoints
    ------------------------"""
    def save_checkpoint(self, checkpoint_path):
        # state_dict: a Python dictionary object that:
        #   - for a model, maps each layer to its parameter tensor;
        #   - for an optimizer, contains info about the optimizers states and hyperparameters used.
        state = {
            'state_dict': self.model.state_dict(),
            'optimizer' : self.optimizer.state_dict()}
        torch.save(state, checkpoint_path)

    def load_checkpoint(self, checkpoint_path):
        state = torch.load(checkpoint_path)
        self.model.load_state_dict(state['state_dict'])
        self.optimizer.load_state_dict(state['optimizer'])
        print('model loaded from %s' % checkpoint_path)
    
    """-----------------------
        Save hyperparameters
    --------------------------"""
    def save_hyperparameters(self):
        """
        model, batch size, epochs, learning rate, loss, optimizer (betas, epsilon)
        """
#         fileNameParameters = create_txt_file_name('./model_loss_and_accuracy/', 'hyperparameters', self.MODEL_NAME)
#         np.savetxt(fileNameParameters, 'Model: {:s}\nBatch size: {:d}\nNum epochs: {:d}\nLearning rate: {:.4e}\nLoss: {:s}\nOptimizer: {:s} \tbetas: {:.4e},{:.4e}   epsilon: {:.4e}'.format(self.MODEL_NAME, self.BATCH_SIZE, self.NUM_EPOCHS, self.LEARNING_RATE,self.LOSS_NAME, self.OPTIM_NAME, self.BETA_1, self.BETA_2, self.EPSILON ))
        
    """---------------------
        Test
    ------------------------"""
    def test(self, current_epoch):
        self.model.eval()  # set evaluation mode
        batch_test_loss = 0.0
        total = 0
        correct = 0
        count = 0
        with torch.no_grad():
            for state, label in self.testSet_loader:
                # Load state to a Torch Variable   
                state, label = state.to(self.device), label.to(self.device)
                # Forward pass only to get logits/output
                output = self.model(state)
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
                test_loss = self.lossCriterion(output, label)
                # Total batch loss
                batch_test_loss += test_loss.item() 
            
        # Average batch loss
        avg_test_loss = batch_test_loss/len(self.testSet_loader)
        # Batch accuracy
        accuracy = 100 * correct / total
        
        # ----- Print accuracy and loss
        # print('[--TEST--] Avg Loss: {:.2e}. Accuracy: {:d}'.format(avg_test_loss, accuracy))

        # ----- Save predictions on last epoch
        if current_epoch == (self.NUM_EPOCHS-1):
            fileNameLabels = create_txt_file_name('./test_set_labels/', 'predictedLabels', self.MODEL_NAME)
            np.savetxt(fileNameLabels, allPredictions, "%i")
        
        return avg_test_loss, accuracy

    """---------------------
        Train
    ------------------------"""
    def train(self):
        self.model.train()  # set training mode
        iteration = 0
        traindat = np.zeros((self.NUM_EPOCHS, 4)) 

        for ep in range(self.NUM_EPOCHS):        
            batch_train_loss = 0.0
            start = time.time()     
            for batch_idx, (data, label) in enumerate(self.trainSet_loader):
               
                # bring data to the computing device, e.g. GPU
                data, label = data.to(self.device), label.to(self.device)
                # forward pass
                output = self.model(data)
                # compute loss: negative log-likelihood
                train_loss = self.lossCriterion(output, label)
                # total epoch training loss
                batch_train_loss += train_loss.item() 
                # backward pass
                # clear the gradients of all tensors being optimized.
                self.optimizer.zero_grad()
                # accumulate (i.e. add) the gradients from this forward pass
                train_loss.backward()
                # performs a single optimization step (parameter update)
                self.optimizer.step()

                # ----- Save checkpoint (binary file): iteration, model, optimizer
                if iteration % self.SAVE_INTERVAL == 0 and iteration > 0:
                    self.save_checkpoint('checkpoints/{:s}/{:s}-{:d}.pth'.format(self.MODEL_NAME, self.MODEL_NAME, iteration)) 

                # ----- Print epoch progress
                # if iteration % self.PRINT_INTERVAL == 0:
                #     print('Epoch: {} [{}/{} ({:.0f}%)]'.format(ep, 
                #         batch_idx * self.BATCH_SIZE, len(self.trainSet_loader.dataset),
                #         100. * batch_idx / len(self.trainSet_loader)))

                iteration += 1
            
            """---------------------
                @end of each epoch
            ------------------------"""
            # Print epoch duration
            end = time.time()
            # print('{:.2f}s'.format(end-start)) 
            # Calculate and print average loss
            avg_train_loss = batch_train_loss/len(self.trainSet_loader)
            print('[--TRAIN--] Avg Loss: {:.2e}'.format(avg_train_loss))    
            # Evaluate model on testSet and save epoch data
            #    - epoch, test_accuracy, train_loss, test_loss
            traindat[ep,0] = ep # current epoch number
            traindat[ep,2] = avg_train_loss
            avg_test_loss, accuracy = self.test(ep) # evaluate model on test set
            traindat[ep,1] = accuracy
            traindat[ep,3] = avg_test_loss 
        
        # ----- Save model accuracy and loss
        fileNameAccuracy = create_txt_file_name('./model_loss_and_accuracy/', 'accuracyAndLoss', self.MODEL_NAME)
        np.savetxt(fileNameAccuracy, traindat, ("%i", "%.d", "%.2e", "%.2e"), 
            header='epoch test_accuracy train_avg_loss test_avg_loss')
        
        # ----- Save final checkpoint 
        self.save_checkpoint('checkpoints/{:s}/{:s}-{:d}.pth'.format(self.MODEL_NAME, self.MODEL_NAME, iteration))


""" --------------------------------------------------------------------------------------
   Main
-----------------------------------------------------------------------------------------"""
if __name__ == "__main__":   
    myNetwork = NeuralNetwork(MODEL_ID)
    myNetwork.save_hyperparameters()
    myNetwork.load_datasets()
    myNetwork.train()
