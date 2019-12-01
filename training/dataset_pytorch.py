import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader

NUM_LABEL_COLUMNS = 1 # switch according to number used in task

class PrimitiveTransitionsSet(Dataset):
    """ 
    Customized dataloader for our primitive transitions dataset 

    """
    def __init__(self, dataList, singleRunFlag=False):
        """ Initialize the dataloader. We transform the np arrays 
        to torch tensors and also floats.

        Input:
          dataList: list of txt files with N rows of 20 elements each: state variables and current primitive 
             (1) state variables (19):
                - time 
                - pos_x pos_y pos_z ori_x ori_y ori_z 
                - vel_x vel_y vel_z angvel_x angvel_y angvel_z 
                - Fx Fy Fz Mx My Mz 
             (2) label (1): Pr0, Pr1, Pr2, Pr3, Pr4 or Pr5 
                 integer corresponding to the highest probability primitive
        """
        if singleRunFlag:
            if len(dataList) != 1:
                print("dataList should be a list of one file name if singleRunFlag = True")
                exit()
        for i, data in enumerate(dataList):
            dataArray = np.loadtxt(data) # dataArray is an (N,20) numpy array
            
            numVars = dataArray.shape[1]
            numDataVectors = dataArray.shape[0]
            numStateVars = numVars - NUM_LABEL_COLUMNS
            
            if i == 0:
                # Associate the state @t with the label @t+1 because we want to learn:
                # what the next primitive should be given the last state observation
                states = dataArray[:-1,0:numStateVars]
                labels = dataArray[1:,-1]
                # Associate the last 5 states with the label @t+1 because we want to learn:
                # what the next primitive should be given the last 5 state observations
                # states = dataArray[:-1,0:numStateVars]
                # labels = dataArray[1:,-1]
            else: 
                single_run_states = dataArray[:-1,0:numStateVars]
                single_run_labels = dataArray[1:,-1]
                states = np.vstack((states, single_run_states))
                labels = np.hstack((labels, single_run_labels))
        
        self.states = torch.from_numpy(states).float()
        self.labels = torch.from_numpy(labels).long()  
        # self.len = numDataVectors - 1 #to account for the @t, @t-1 shift 
        self.len = states.shape[0]
        print('DATA SIZES:\n  States: {:d}\n  Labels: {:d}\n  Length:{:d}'.format(states.shape[0], labels.shape[0], self.len))
        # exit()
    
    def __getitem__(self, index):
        """
        Get a sample from the dataset
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
    