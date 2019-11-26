import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader

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
             (2) labels (5): Pr0 Pr1 Pr2 Pr3 Pr4 (current primitive probabilities)
        """
        dataArray = np.loadtxt(data) # dataArray is an (N,24) numpy array
        
        numVars = dataArray.shape[1]
        numDataVectors = dataArray.shape[0]
        numPrimitives = 5
        numStateVars = numVars - numPrimitives
        
        states = dataArray[:,0:numStateVars]
        labels = dataArray[:,numStateVars:]
        
        self.states = torch.from_numpy(states).float()
        self.labels = torch.from_numpy(labels).float()  
        self.len = numDataVectors
    
    def __getitem__(self, index):
        """ Get a sample from the dataset
        """
        image = self.states[index]
        label = self.labels[index]

        # return image and label
        return image, label

    def __len__(self):
        """
        Total number of samples in the dataset
        """
        return self.len
    