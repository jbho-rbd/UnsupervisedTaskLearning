import itertools
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import rc
rc('text',usetex=True)
from sklearn import svm, datasets
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
from copy import copy
def plot_confusion_matrix(cm, title='Confusion matrix', cmap=plt.cm.Oranges,normalize=False):
    cm_abs = copy(cm)
    for i in range(cm.shape[0]):
        cm[i,:] /= np.sum(cm[i,:])
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    # plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(cm.shape[0])
    plt.xticks(tick_marks, rotation=45)
    ax = plt.gca()
    ax.set_xticklabels((('none', 'fsm', 'align','engage','screw','tighten')))
    ax.set_yticklabels(('none', 'fsm', 'align','engage','screw','tighten'))
    plt.yticks(tick_marks)
    plt.axis([-0.5, cm.shape[0]-0.5, -0.5, cm.shape[0]-0.5])

    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, "{0:.2f}\n({1:.0f})".format(cm[i, j], cm_abs[i,j]),
                 horizontalalignment="center",
                 verticalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.ylabel('Manual Label')
    plt.xlabel('Automatic Label')
    plt.tight_layout()

cm = np.loadtxt('failcount_final.dat')
# np.set_printoptions(precision=1) #!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
print('Confusion matrix, without normalization')
print(cm)
fig, ax = plt.subplots()
plot_confusion_matrix(cm,normalize=False)
plt.savefig('cm.png',dpi=600)
plt.show()