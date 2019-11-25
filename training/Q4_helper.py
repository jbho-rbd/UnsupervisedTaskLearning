import numpy as np
from PIL import Image

def load_dataset(label_file, folder):
    labels = np.load(label_file)
    # labels = labels[:10]
    num_images = labels.shape[0]
    images = np.empty(shape=(num_images, 480, 640,3))
    for i in range(num_images):
        im = Image.open(folder+'/img_%03d.png'%(i))
        images[i] = np.array(im)
    return images, labels

def load_training_dataset(label_file, folder):
    labels = np.load(label_file)
    # labels = labels[:10]
    num_images = labels.shape[0]
    images = np.empty(shape=(num_images, 480, 640,3))
    for i in range(num_images):
        im = Image.open(folder+'/img_%03d_train.png'%(i))
        images[i] = np.array(im)
    return images, labels


def load_testing_dataset(label_file, folder):
    labels = np.load(label_file)
    # labels = labels[:10]
    num_images = labels.shape[0]
    images = np.empty(shape=(num_images, 480, 640,3))
    for i in range(num_images):
        im = Image.open(folder+'/img_%03d_test.png'%(i))
        images[i] = np.array(im)
    return images, labels   