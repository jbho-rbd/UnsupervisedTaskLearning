import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader

NUM_LABEL_COLUMNS = 1 # switch according to number used in task

class PrimitiveTransitionsSet(Dataset):
    """ 
    Customized dataloader for our primitive transitions dataset 

    """
    def __init__(self, data):
        """ Initialize the dataloader. We transform the np arrays 
        to torch tensors and also floats.

        Input:
          data: txt file with N rows of 24 elements each: state variables and current primitive probabilities
             (1) state variables (19):
                - time 
                - pos_x pos_y pos_z ori_x ori_y ori_z 
                - vel_x vel_y vel_z angvel_x angvel_y angvel_z 
                - Fx Fy Fz Mx My Mz 
             (2) labels (6): Pr0 Pr1 Pr2 Pr3 Pr4 Pr5 (current primitive probabilities)
        """
        dataArray = np.loadtxt(data) # dataArray is an (N,24) numpy array
        
        numVars = dataArray.shape[1]
        numDataVectors = dataArray.shape[0]
        numStateVars = numVars - NUM_LABEL_COLUMNS
        
        # Associate the state @t with the label @t+1 because that's what we want to learn
        states = dataArray[:-1,0:numStateVars]
        labels = dataArray[1:,-1]
        
        self.states = torch.from_numpy(states).float()
        self.labels = torch.from_numpy(labels).long()  
        self.len = numDataVectors - 1 #to account for the @t, @t-1 shift
    
    def __getitem__(self, index):
        """ Get a sample from the dataset
        """
        state = self.states[index]
        label = self.labels[index]

        # return state and label
        return state, label

    def __len__(self):
        """
        Total number of samples in the dataset
        """
        return self.len
    