import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from scipy.spatial.transform import Rotation as R
from matplotlib import rc
from read_data import read_data0, read_data1
from classifier import Pr, observation_tests
import scipy
import pylab

def print_stats(name, x):
    mean = np.mean(x)
    stdev = np.std(x)
    print("{0:s} Mean: {1:f}, StDev: {2:f}".format(name, mean/(1-mean),stdev))
    # TS, p = scipy.stats.shapiro(x)
    # print("p_value: {0:e}, shapiro test".format(p))


def print_stats_xyz(name, x):
    print_stats(name + " x", x[:,0])
    print_stats(name + " y", x[:,1])
    print_stats(name + " z", x[:,2])
    # print_stats(name + " norm", np.linalg.norm(x,axis=1))


#y axis becomes z
#z axis becomes x
#x axis becomes y
# t, p1, euler, omegas, F, M = read_data0('../data/run1.dat', '../data/bias.force')
tlabels = np.genfromtxt("../data2/run1_tlabels",dtype=float)
tlabels = np.insert(tlabels,0,0.0)
# vlines=np.array([0.0, 1.08, 3.27,4.2,5.5,6.72,6.9,7.44, 7.6, 8.02, 8.21, 8.68, 8.98, 10.5])
# np.savetxt("../data2/run1_tlabels",vlines[1:])
labels=[Pr(int(idx)) for idx in np.genfromtxt("../data2/run1_prmlabels")]
n_primitives=5
for prim in [Pr.none, Pr.fsm, Pr.contact, Pr.align, Pr.screw]:
    tpairs = []
    for i in range(len(labels)):#collect different labels and time periods corresponding to this primitive
        if(labels[i] == prim):
            tpairs.append([tlabels[i],tlabels[i+1]])
    print("Primitive: {0:s}".format(Pr(prim)))
    t, obs = read_data1('../data2/run1', '../data2/bias.force',output_fmt='obs',tpairlist=tpairs)
    # print(tpairs)
    ntests=6
    p = np.zeros((len(t),ntests))
    for i in range(len(t)):
        p[i] = observation_tests(obs[i])
    testname=("vel = 0", "F = 0 ", "M = 0", "ang_vel = 0","ang_vel_z = 0", "M_z = 0")
    for i in [0]:#range(ntests):
        print_stats("p "+testname[i] + ": ",p[:,i])
    # t, p1, vels, euler, omegas, F, M = read_data1('../data2/run1', '../data2/bias.force',tpairlist=tpairs)
    # print_stats_xyz("velocity", vels)
    # plt.title(str(Pr(prim))+" M_z")
    # plt.hist(M[:,2],bins=200)
    # plt.show()
    # scipy.stats.probplot(omegas[:,2], dist=scipy.stats.laplace, plot=pylab)
    # pylab.title(str(Pr(prim))+"omega_z")
    # pylab.show()
    # plt.show()
    # print_stats_xyz("omegas", omegas)
    # print_stats_xyz("forces", F)
    # print_stats_xyz("moments", M)


