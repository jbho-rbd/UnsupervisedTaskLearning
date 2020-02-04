import matplotlib.pyplot as plt
import numpy as np
import argparse
from matplotlib import rc, rcParams
from scipy.special import softmax
# rc('text',usetex=True)

#rcParams['axes.titlesize'] = 'x-large'
#rcParams['axes.labelsize'] = 'x-large'
#rcParams['xtick.labelsize'] = 'x-large'
#rcParams['ytick.labelsize'] = 'x-large'
#rcParams['legend.fontsize'] = 'large'

""" --------------------------------------------------------------------------------------
   Command Line Arguments: Network Selection
-----------------------------------------------------------------------------------------"""
parser = argparse.ArgumentParser(description='Network Plots')
parser.add_argument('--f1', required = True, type=str, metavar='FILE_NAME1',
                    help='pass in filename containing data')
parser.add_argument('--f2', required = False, type=str, metavar='FILE_NAME2',
                    help='pass in filename containing data')
parser.add_argument('--f3', required = False, type=str, metavar='FILE_NAME3',
                    help='pass in filename containing data')
args = parser.parse_args()
FILE_NAME1 = args.f1
FILE_NAME2 = args.f2
FILE_NAME3 = args.f3

""" --------------------------------------------------------------------------------------
    Plots
-----------------------------------------------------------------------------------------"""
def create_fig_file_name(folder_path, description_string, data_file_name):
    figname = folder_path + description_string + '_' + data_file_name + ".png"
    return figname 

def plot_learning_curves(f1,f2,f3):
    f_loss,ax_loss = plt.subplots()
    f_loss.set_tight_layout(True)
    f_accuracy,ax_accuracy = plt.subplots()
    f_accuracy.set_tight_layout(True)
    plt.tight_layout()
    nEpochs = 40
    names = ['Logistic Regression', '2 Layer', '2 Layer Regu + history']
    color = ['C0', 'C1', 'C2']

    for i, filename in enumerate([f1, f2, f3]):
        data = np.loadtxt('./final_figures/' + filename + '.txt')
        epoch = data[:nEpochs,0]
        test_accuracy = data[:nEpochs,1]
        training_loss = data[:nEpochs,2]
        test_loss = data[:nEpochs,3]
        training_accuracy = data[:nEpochs,4]

        ax_loss.plot(epoch, training_loss, color[i]+':', label=names[i] + ': training')
        ax_loss.plot(epoch, test_loss, color[i]+'-',label=names[i] + ': test')
        ax_loss.set_yscale('log')
        ax_loss.set_xlabel('Epoch \#')
        ax_loss.set_ylabel('Cross Entropy Loss')
        #ax_loss.set_yticks(10**np.arange(-1,0.1))
        ax_loss.set_yticks([0.2, 0.5, 1.0, 2.0])
        #print(np.arange(-1,0,0.25))
        ax_loss.legend(loc = 'upper right')

        ax_accuracy.plot(epoch, training_accuracy, color[i]+':', label=names[i] + ': train')
        ax_accuracy.plot(epoch, test_accuracy, color[i]+'-', label=names[i] + ': test')
        ax_accuracy.set_xlabel('Epoch \#')
        ax_accuracy.set_ylabel('Accuracy (\%)')
        ax_accuracy.legend(loc = 'lower right')

    ax_loss.yaxis.set_major_formatter(plt.ScalarFormatter())
    figureName = create_fig_file_name('./final_figures/', 'LOSS_FIGURE', f1 + f2 + f3)
    f_loss.savefig(figureName, dpi = 600)
    figureName = create_fig_file_name('./final_figures/', 'ACCURACY_FIGURE', f1 + f2 + f3)
    f_accuracy.savefig(figureName, dpi = 600)
    plt.show()

def plot_primitive_probabilities(filename):
    prob_data = np.loadtxt('./output_probabilities/' + filename + '.txt')
    prob_data_normalized = softmax(prob_data,axis = 1)
    num_time_steps = prob_data.shape[0]
    print(num_time_steps)
    step = 0.025 #sensor data was sampled at 40Hz
    time_data = np.arange(0,(num_time_steps-0.5)*step, step)
    plt.plot(time_data, prob_data_normalized[:,0], label = "none")
    plt.plot(time_data, prob_data_normalized[:,1], label = "fsm")
    plt.plot(time_data, prob_data_normalized[:,2], label = "align")
    plt.plot(time_data, prob_data_normalized[:,3], label = "engage")
    plt.plot(time_data, prob_data_normalized[:,4], label = "screw")
    plt.plot(time_data, prob_data_normalized[:,5], label = "tigthen")
    figureName = create_fig_file_name('./final_figures/', 'OUTPUT_PROBS', filename)
    plt.legend()
    plt.xlabel("time")
    plt.ylabel("probability")
    plt.savefig(figureName, dpi = 600)
    plt.show()
""" --------------------------------------------------------------------------------------
   Main
-----------------------------------------------------------------------------------------"""
if __name__ == "__main__":   
    # plot_learning_curves(FILE_NAME1, FILE_NAME2, FILE_NAME3)
    plot_primitive_probabilities(FILE_NAME1)
