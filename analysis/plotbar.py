import matplotlib.pyplot as plt
import numpy as np
from classifier import Pr
from matplotlib import rc, rcParams
rc('text',usetex=True)
rcParams['axes.titlesize'] = 'x-large'
rcParams['axes.labelsize'] = 'large'
rcParams['xtick.labelsize'] = 'large'
rcParams['ytick.labelsize'] = 'large'
faildat = np.genfromtxt('failcount_final.dat')
faildat = faildat/np.sum(np.sum(faildat))
N = faildat.shape[0]
ind = range(N)
width=0.85
base = np.zeros(N)
plt.figure(figsize=(3,5.3))
for i in ind:
	plt.bar(ind,faildat[:,i],width,bottom=base,label=str(Pr(i))[3:])
	base = base + faildat[:,i]
xlabels = [str(Pr(i))[3:] for i in ind]
plt.xticks(ind,xlabels,rotation=50)
plt.ylabel('Auto Label: Rate of Occurence')
plt.xlabel('Manual Label')
plt.legend()
plt.tight_layout()
plt.savefig('/Users/jonathanho/Downloads/bar.png',dpi=600)
plt.show()
