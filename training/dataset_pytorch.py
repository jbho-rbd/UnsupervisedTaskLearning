import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader


class BallTrackSet(Dataset):
    """ 
    A customized dataloader for our ball tracking dataset 

    """
    def __init__(self, images, labels):
        """ Initialize the dataloader. We transform the np arrays 
        to torch tensors and also floats 
        Input:
          images: (N,480,640,3) numpy array of the images
          labels: (N,3) numpy array of the ball location labels
        """


        total_images = images.transpose((0,3,1,2))
        total_images /= 255.0 
        self.total_images = torch.from_numpy(total_images).float()
        self.total_labels = torch.from_numpy(labels).float()
        
        self.len = total_images.shape[0]
    
    def __getitem__(self, index):
        """ Get a sample from the dataset
        """
        image = self.total_images[index]
        label = self.total_labels[index]

        # return image and label
        return image, label

    def __len__(self):
        """
        Total number of samples in the dataset
        """
        return self.len
    