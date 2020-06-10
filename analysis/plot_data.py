import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from scipy.spatial.transform import Rotation as R
from matplotlib import rc
from read_data import read_data0, read_data1
from classifier import Pr
import sys

def plot_file(file,tlabelfile=None,prlabelfile=None):
    #y axis becomes z
    #z axis becomes x
    #x axis becomes y
    """
    Inputs: 
        file: filename of raw data collected using the redis logger
        tlabelfile: text file with the endtime of each primitive
        prlabelfile: text file with list of primitives as an integer 

    """    
    t, p1, vels, euler, omegas, F, M = read_data1(file, '../data/medium_cap/raw_medium_cap/bias.force')#,t0=0,t1=5.0)
    """
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    quiverlength = 0.03
    poses_sampled = poses[0::50]
    ax.quiver(poses_sampled[:,0,3], poses_sampled[:,1,3], poses_sampled[:,2,3],
              poses_sampled[:,0,0], poses_sampled[:,1,0], poses_sampled[:,2,0],
              color='r', length=quiverlength, arrow_length_ratio=0.05)
    ax.quiver(poses_sampled[:,0,3], poses_sampled[:,1,3], poses_sampled[:,2,3],
              poses_sampled[:,0,1], poses_sampled[:,1,1], poses_sampled[:,2,1],
              color='b', length=quiverlength, arrow_length_ratio=0.05)
    ax.quiver(poses_sampled[:,0,3], poses_sampled[:,1,3], poses_sampled[:,2,3],
              poses_sampled[:,0,2], poses_sampled[:,1,2], poses_sampled[:,2,2],
              color='g', length=quiverlength, arrow_length_ratio=0.05)
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_zlabel('z')
    """
    f,ax=plt.subplots(7,1,sharex=True,figsize=(8,9))
    ax[0].plot(t,p1[:,0],'r',label='x')
    ax[0].plot(t,p1[:,1],'g',label='y')
    ax[0].plot(t,p1[:,2],'b',label='z')
    ax[0].set_ylabel('meters')
    ax[0].legend(loc='upper right')
    ax[1].plot(t,vels[:,0],'r',label='x')
    ax[1].plot(t,vels[:,1],'g',label='y')
    ax[1].plot(t,vels[:,2],'b',label='z')
    ax[1].set_ylabel('meters/s')
    ax[1].legend(loc='upper right')
    ax[2].plot(t,np.pi/180*euler[:,2],'r',label='$\\theta$')
    ax[2].plot(t,np.pi/180*euler[:,1],'g',label='$\Phi$')
    ax[2].plot(t,np.pi/180*euler[:,0],'b',label='$\Psi$')
    ax[2].set_ylabel('radians')
    ax[2].legend(loc='upper right')
    ax[3].plot(t,omegas[:,0],'r',label='$\omega_x$')
    ax[3].plot(t,omegas[:,1],'g',label='$\omega_y$')
    ax[3].plot(t,omegas[:,2],'b',label='$\omega_z$')
    ax[3].set_ylabel('rad/s')
    ax[3].legend(loc='upper right')
    ax[4].plot(t,F[:,0],'r',label='$f_x$')
    ax[4].plot(t,F[:,1],'g',label='$f_y$')
    ax[4].plot(t,F[:,2],'b',label='$f_z$')
    ax[4].set_ylabel('N')
    ax[4].legend(loc='upper right')
    ax[5].plot(t,M[:,0],'r',label='$\\tau_x$')
    ax[5].plot(t,M[:,1],'g',label='$\\tau_y$')
    ax[5].plot(t,M[:,2],'b',label='$\\tau_z$')
    ax[5].set_ylabel('Nm')
    ax[5].legend(loc='upper right')
    plotLabels=tlabelfile is not None and prlabelfile is not None
    if plotLabels:
        vlines = np.genfromtxt(tlabelfile,dtype=float)
        vlines = np.insert(vlines,0,0.0)
        labels=[Pr(int(idx)) for idx in np.genfromtxt(prlabelfile)]
        for i in range(7):
            for vline in vlines[1:]:
                ax[i].axvline(x=vline,color='k',linestyle=':')
        y = 0.5
        xcords = 0.5*(vlines[1:] + vlines[:-1])
        for i in range(len(xcords)):
            ax[6].text(xcords[i],y,labels[i],horizontalalignment='center',rotation=90, verticalalignment='center')
        ax[6].set_xlabel('time')
        ax[6].get_yaxis().set_ticks([])
        labels=('A','B','C','D','E','F')
        for i in range(6):
            secaxy = ax[i].twinx()
            secaxy.set_ylabel(labels[i],rotation=0,labelpad=10)
            secaxy.get_yaxis().set_ticks([])
        f.align_ylabels(ax)

def getlabels(likelihoodfile, tlabelFile = None, prlabelFile = None):
    """
    Inputs:
        likelihoodfile: filename of a text data file that contains likelihoods in the following format:
        # time p[s0] p[s1] p[s2] p[s3] p[s4] ...
        where s0, s1, are the different primitives

    Outputs:
        tlabelFile: a file of the times t at whjich the primitive changes (includes the t_final)
        prlabelFile: a file of the list of primitives

    these two outputs can be fed to plot_file

    """
    dat = np.genfromtxt(likelihoodfile)
    t = dat[:,0]
    likelihoods = dat[:,1:]
    prs = np.argmax(likelihoods,axis=1)
    tlist = []
    prlist = [prs[0]]
    for i in range(len(t)-1):
        if prs[i+1] != prs[i]:
            prlist.append(prs[i+1])
            tlist.append(t[i+1])
    tlist.append(t[-1])
    np.savetxt(tlabelFile,tlist)
    np.savetxt(prlabelFile,prlist)
    # return tlist, prlist, prs
    
def compute_success_rate(likelihoodfile, tlabelFile_groundTruth, prlabelFile_groundTruth):
    dat = np.genfromtxt(likelihoodfile)
    t = dat[:,0]
    likelihoods = dat[:,1:]
    prs = np.argmax(likelihoods,axis=1)
    successes = 0.0
    tlabels=np.genfromtxt(tlabelFile_groundTruth)
    prlabels=np.genfromtxt(prlabelFile_groundTruth)
    pr0 = prlabels[0]
    pr1 = -1
    margin = 0.05
    pr_idx = 0
    count = 0
    for i, t_i in enumerate(t):
        if t_i > tlabels[-1]:
            break
        if pr1 < 0 and t_i + margin > tlabels[pr_idx]:
            if pr_idx < len(tlabels) - 1:
                pr1 = int(prlabels[pr_idx + 1])
        elif t_i - margin > tlabels[pr_idx]:
            pr0 = pr1
            pr1 = -1
            pr_idx = pr_idx + 1
        successes += int(prs[i] == pr0 or prs[i] == pr1)
        count += 1
    return successes / count


if __name__ == "__main__":
    run_number=int(sys.argv[1])
    #---making labels---
    # vlines=np.array([0.0, 0.84, 3.27, 4.21, 5.07, 6.74, 6.93, 7.44, 7.588, 8.06, 8.20, 8.68, 8.89, 9.56, 10.36])
    # np.savetxt("../data/medium_cap/raw_medium_cap/run{0:d}_tlabels".format(run_number),vlines[1:])
    # labels = [Pr.none, Pr.fsm, Pr.align, Pr.engage, Pr.screw,  Pr.none, Pr.screw, Pr.none, Pr.screw, Pr.none, Pr.screw, Pr.none, Pr.screw, Pr.tighten]
    # np.savetxt("../data/medium_cap/raw_medium_cap/run{0:d}_prmlabels".format(run_number),[label.value for label in labels])
    # plot_file('../data/medium_cap/raw_medium_cap/run{0:d}'.format(run_number))
    # plot_file('../data/medium_cap/raw_medium_cap/run{0:d}'.format(run_number),tlabelfile="../data/medium_cap/raw_medium_cap/run{0:d}_tlabels".format(run_number),prlabelfile="../data/medium_cap/raw_medium_cap/run{0:d}_prmlabels".format(run_number))
    
    #--- saving tlabels and prmlabels from likelihoods files  ----
    # dummya, dummyb, prs = getlabels("results/run{0:d}_likelihoods".format(run_number), tlabelFile="results/run{0:d}_tlabels".format(run_number), prlabelFile="results/run{0:d}_prmlabels".format(run_number))
    getlabels("results/run{0:d}_likelihoods".format(run_number), tlabelFile="results/run{0:d}_tlabels".format(run_number), prlabelFile="results/run{0:d}_prmlabels".format(run_number))
    # time, X = read_data1('../data/medium_cap/raw_medium_cap/run' + str(run_number), '../data/medium_cap/raw_medium_cap/bias.force',output_fmt='array')
    # N = len(time)
    # headerstr = "time pos_x pos_y pos_z ori_x ori_y ori_z vel_x vel_y vel_z angvel_x angvel_y angvel_z Fx Fy Fz Mx My Mz Pr"
    # np.savetxt("../data/medium_cap/raw_medium_cap/run{0:d}_labelled".format(run_number),np.hstack((np.reshape(time,(N,1)), X, np.reshape(prs,(N,1)))),header=headerstr)

    success_rate = compute_success_rate("results/run{0:d}_likelihoods".format(run_number), "../data/medium_cap/raw_medium_cap/run{0:d}_tlabels".format(run_number),"../data/medium_cap/raw_medium_cap/run{0:d}_prmlabels".format(run_number))
    print("success_rate: {0:f}".format(success_rate))
    
    #---plotting
    # Plot labelled run 
    # plot_file('../data/medium_cap/raw_medium_cap/run{0:d}'.format(run_number),tlabelfile="results/run{0:d}_tlabels".format(run_number),prlabelfile="results/run{0:d}_prmlabels".format(run_number))
    # Plot labelled run 1
    # plot_file('../data/medium_cap/raw_medium_cap/run1'.format(run_number),tlabelfile="../data/medium_cap/raw_medium_cap/run1_tlabels".format(run_number),prlabelfile="../data/medium_cap/raw_medium_cap/run1_prmlabels".format(run_number))
    # plt.savefig("results/labelled_run{0:d}.png".format(run_number),dpi=600)
    # plt.show()
