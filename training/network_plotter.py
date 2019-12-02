import matplotlib.pyplot as plt
import numpy as np
import argparse

""" --------------------------------------------------------------------------------------
   Command Line Arguments: Network Selection
-----------------------------------------------------------------------------------------"""
parser = argparse.ArgumentParser(description='Network Plots')
parser.add_argument('--filename', required = True, type=str, metavar='FILE_NAME',
                    help='pass in filename containing data')
args = parser.parse_args()
FILE_NAME = args.filename 

""" --------------------------------------------------------------------------------------
    Plots
-----------------------------------------------------------------------------------------"""
def create_fig_file_name(folder_path, description_string, data_file_name):
    figname = folder_path + description_string + '_' + data_file_name + ".png"
    return figname 

def plot_learning_curves(filename):
    data = np.loadtxt('model_loss_and_accuracy/' + filename + '.txt')
    epoch = data[:,0]
    test_accuracy = data[:,1]
    training_loss = data[:,2]
    test_loss = data[:,3]

    plt.figure()
    plt.plot(epoch, training_loss)
    plt.plot(epoch, test_loss)
    plt.yscale('log')
    plt.xlabel('Epoch #')
    plt.ylabel('Loss')
    plt.legend(['training','test'], loc = 'upper right')
    figureName = create_fig_file_name('./model_loss_and_accuracy/', 'LOSS_FIGURE', filename)
    plt.savefig(figureName, dpi = 600)

    plt.figure()
    plt.plot(epoch, test_accuracy)
    plt.xlabel('Epoch #')
    plt.ylabel('Test Accuracy (%)')
    figureName = create_fig_file_name('./model_loss_and_accuracy/', 'ACCURACY_FIGURE', filename)
    plt.savefig(figureName, dpi = 600)
    plt.show()

""" --------------------------------------------------------------------------------------
   Main
-----------------------------------------------------------------------------------------"""
if __name__ == "__main__":   
    plot_learning_curves(FILE_NAME)
