"""======================================================================================
 plot_data.py
 
 Functions: 
    - plot_file
    - getlabels
    - compute_success_rate
    - write_Pr_file

 Main: computes the success rate for each run
 
Last update, Fall 2020
======================================================================================"""
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from scipy.spatial.transform import Rotation as R
from matplotlib import rc, rcParams
from read_data import read_data1
from classifier import Pr
import sys
from shutil import copyfile

def plot_file(file,tlabelfile=None,prlabelfile=None,tlabelfileTruth=None,prlabelfileTruth=None,
    plot_pos=True,plot_vel=True,plot_ori=True,plot_ang_vel=True,plot_force=True,plot_moment=True):
    """
    # y axis becomes z
    # z axis becomes x
    # x axis becomes y
    
    Inputs: 
        - file: filename of raw data collected using the redis logger
        - tlabelfile: text file with the endtime of each primitive
        - prlabelfile: text file with list of primitives as an integer 

    """    
    t, p1, vels, euler, omegas, F, M = read_data1(file, '../data/medium_cap/raw_medium_cap/bias.force')#,t0=0,t1=5.0)
     
    # Figure: Frames representing the cap trajectory and pose
    # fig = plt.figure()
    # ax = fig.add_subplot(111, projection='3d')
    # quiverlength = 0.03
    # poses_sampled = poses[0::50]
    # ax.quiver(poses_sampled[:,0,3], poses_sampled[:,1,3], poses_sampled[:,2,3],
    #           poses_sampled[:,0,0], poses_sampled[:,1,0], poses_sampled[:,2,0],
    #           color='r', length=quiverlength, arrow_length_ratio=0.05)
    # ax.quiver(poses_sampled[:,0,3], poses_sampled[:,1,3], poses_sampled[:,2,3],
    #           poses_sampled[:,0,1], poses_sampled[:,1,1], poses_sampled[:,2,1],
    #           color='b', length=quiverlength, arrow_length_ratio=0.05)
    # ax.quiver(poses_sampled[:,0,3], poses_sampled[:,1,3], poses_sampled[:,2,3],
    #           poses_sampled[:,0,2], poses_sampled[:,1,2], poses_sampled[:,2,2],
    #           color='g', length=quiverlength, arrow_length_ratio=0.05)
    # ax.set_xlabel('x')
    # ax.set_ylabel('y')
    # ax.set_zlabel('z')
    
    nPlots = np.sum([int(flag) for flag in [plot_pos, plot_vel, plot_ori, plot_ang_vel, plot_force, plot_moment]])
    plotTruthLabels=tlabelfileTruth is not None and prlabelfileTruth is not None
    if plotTruthLabels:
        nTotalPlots =  nPlots + 2
    else:
        nTotalPlots =  nPlots + 1
    f,ax=plt.subplots(nTotalPlots,1,sharex=True,figsize=(8,9))
    fig_idx = 0
    if plot_pos:
        ax[fig_idx].plot(t,p1[:,0],'r',label='x')
        ax[fig_idx].plot(t,p1[:,1],'g',label='y')
        ax[fig_idx].plot(t,p1[:,2],'b',label='z')
        ax[fig_idx].set_ylabel('meters')
        ax[fig_idx].legend(loc='upper right')
        fig_idx += 1
    if plot_vel:
        ax[fig_idx].plot(t,vels[:,0],'r',label='x')
        ax[fig_idx].plot(t,vels[:,1],'g',label='y')
        ax[fig_idx].plot(t,vels[:,2],'b',label='z')
        ax[fig_idx].set_ylabel('meters/s')
        ax[fig_idx].legend(loc='upper right')
        fig_idx += 1
    if plot_ori:
        ax[fig_idx].plot(t,np.pi/180*euler[:,2],'r',label='$\\theta$')
        ax[fig_idx].plot(t,np.pi/180*euler[:,1],'g',label='$\Phi$')
        ax[fig_idx].plot(t,np.pi/180*euler[:,0],'b',label='$\Psi$')
        ax[fig_idx].set_ylabel('radians')
        ax[fig_idx].legend(loc='upper right')
        fig_idx += 1
    if plot_ang_vel:
        ax[fig_idx].plot(t,omegas[:,0],'r',label='$\omega_x$')
        ax[fig_idx].plot(t,omegas[:,1],'g',label='$\omega_y$')
        ax[fig_idx].plot(t,omegas[:,2],'b',label='$\omega_z$')
        ax[fig_idx].set_ylabel('rad/s')
        ax[fig_idx].legend(loc='upper right')
        fig_idx += 1
    if plot_force:
        ax[fig_idx].plot(t,F[:,0],'r',label='$f_x$')
        ax[fig_idx].plot(t,F[:,1],'g',label='$f_y$')
        ax[fig_idx].plot(t,F[:,2],'b',label='$f_z$')
        ax[fig_idx].set_ylabel('N')
        ax[fig_idx].legend(loc='upper right')
        fig_idx += 1
    if plot_moment:
        ax[fig_idx].plot(t,M[:,0],'r',label='$\\tau_x$')
        ax[fig_idx].plot(t,M[:,1],'g',label='$\\tau_y$')
        ax[fig_idx].plot(t,M[:,2],'b',label='$\\tau_z$')
        ax[fig_idx].set_ylabel('Nm')
        ax[fig_idx].legend(loc='upper right')
    
    plotLabels=tlabelfile is not None and prlabelfile is not None
    if plotLabels:
        vlines = np.genfromtxt(tlabelfile,dtype=float)
        vlines = np.insert(vlines,0,0.0)
        labels=[Pr(int(idx)) for idx in np.genfromtxt(prlabelfile)]
        for i in range(nPlots + 1):
            for vline in vlines[1:]:
                ax[i].axvline(x=vline,color='k',linestyle=':')
        y = 0.5
        xcords = 0.5*(vlines[1:] + vlines[:-1])
        for i in range(len(xcords)):
            ax[nPlots].text(xcords[i],y,str(labels[i])[3:],horizontalalignment='center',rotation=90, verticalalignment='center',size='x-large')
        if not plotTruthLabels:
            ax[nPlots].set_xlabel('time')
        ax[nPlots].get_yaxis().set_ticks([])
        ax[nPlots].set_facecolor('#e6b8afff')
        labels=('A','B','C','D','E','F')
        for i in range(nPlots):
            secaxy = ax[i].twinx()
            secaxy.set_ylabel(labels[i],rotation=0,labelpad=10)
            secaxy.get_yaxis().set_ticks([])
        f.align_ylabels(ax[:nPlots])
    if plotTruthLabels:
        vlines = np.genfromtxt(tlabelfileTruth,dtype=float)
        vlines = np.insert(vlines,0,0.0)
        labels=[Pr(int(idx)) for idx in np.genfromtxt(prlabelfileTruth)]
        for vline in vlines[1:]:
            ax[nPlots + 1].axvline(x=vline,color='k',linestyle=':')
        y = 0.5
        xcords = 0.5*(vlines[1:] + vlines[:-1])
        for i in range(len(xcords)):
            ax[nPlots + 1].text(xcords[i],y,str(labels[i])[3:],horizontalalignment='center',rotation=90, verticalalignment='center',size='x-large')
        ax[nPlots + 1].set_xlabel('time')
        ax[nPlots + 1].get_yaxis().set_ticks([])
        ax[nPlots].set_ylabel('Auto \n Labelled',rotation=90,labelpad=4)
        ax[nPlots + 1].set_ylabel('Manually \n Labelled',rotation=90,labelpad=4)
    # ax[0].set_title('Demonstration \#2: Primitive Labelling Success Rate: 93\%')

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

def compute_success_rate(likelihoodfile, tlabelFile_groundTruth, prlabelFile_groundTruth, failureFile = None):
    dat = np.genfromtxt(likelihoodfile)
    t = dat[:,0]
    likelihoods = dat[:,1:]
    prs = np.argmax(likelihoods,axis=1)
    successes = 0.0
    tlabels=np.genfromtxt(tlabelFile_groundTruth)
    prlabels=np.genfromtxt(prlabelFile_groundTruth)
    pr0 = int(prlabels[0])
    pr1 = -1
    margin = 0.05
    pr_idx = 0
    count = 0
    pr_actual = int(prlabels[0])
    pr_idx_actual = 0
    countFailures = failureFile is not None
    if countFailures:
        failure_count = np.genfromtxt(failureFile,dtype=int)
    for i, t_i in enumerate(t):
        if t_i > tlabels[-1]:
            break
        if t_i > tlabels[pr_idx_actual]:
            pr_actual = int(prlabels[pr_idx_actual + 1])
            pr_idx_actual += 1
        if pr1 < 0 and t_i + margin > tlabels[pr_idx]:
            if pr_idx < len(tlabels) - 1:
                pr1 = int(prlabels[pr_idx + 1])
        elif t_i - margin > tlabels[pr_idx]:
            pr0 = pr1
            pr1 = -1
            pr_idx = pr_idx + 1
        success = (prs[i] == pr0 or prs[i] == pr1)
        if countFailures:
            failure_count[pr_actual,prs[i]] += 1
        successes += int(success)
        count += 1
    if countFailures:
        np.savetxt(failureFile,failure_count,fmt="%i")
    return successes / count
    
def write_Pr_file(t,X, tlabelFile_groundTruth, prlabelFile_groundTruth):
    prs = np.zeros(len(t))
    tlabels=np.genfromtxt(tlabelFile_groundTruth)
    prlabels=np.genfromtxt(prlabelFile_groundTruth)
    current_pr = prlabels[0]
    pr_idx = 0
    N = len(t)
    for i, t_i in enumerate(t):
        if t_i > tlabels[-1]:
            N = i
            break
        if t_i > tlabels[pr_idx]:
            current_pr = int(prlabels[pr_idx + 1])
            pr_idx += 1
        prs[i] = current_pr
    return np.hstack((np.reshape(t[:N],(N,1)), X[:N], np.reshape(prs[:N],(N,1))))

""" --------------------------------------------------------------------------------------
   MAIN
-----------------------------------------------------------------------------------------"""
# if __name__ == "__main__":
    # run_number=int(sys.argv[1])
    # failFileName = 'results/failcount.txt' #SEE ME
    # copyfile('results/failcount0.txt',failFileName) #SEE ME
    # np.savetxt(failFileName, np.zeros((6,6),dtype=int))
    
    # for run_number in range(1, 19): #SEE ME
    #     if run_number == 11 or run_number == 16: #SEE ME
    #         continue #SEE ME

    #--- making labels: tlabels and prmlabels ---
    # vlines=np.array([0.0, 0.84, 3.27, 4.21, 5.07, 6.74, 6.93, 7.44, 7.588, 8.06, 8.20, 8.68, 8.89, 9.56, 10.36])
    # np.savetxt("../data/medium_cap/raw_medium_cap/run{0:d}_tlabels".format(run_number),vlines[1:])
    # labels = [Pr.none, Pr.fsm, Pr.align, Pr.engage, Pr.screw,  Pr.none, Pr.screw, Pr.none, Pr.screw, Pr.none, Pr.screw, Pr.none, Pr.screw, Pr.tighten]
    # np.savetxt("../data/medium_cap/raw_medium_cap/run{0:d}_prmlabels".format(run_number),[label.value for label in labels])
    # plot_file('../data/medium_cap/raw_medium_cap/run{0:d}'.format(run_number))
    # plot_file('../data/medium_cap/raw_medium_cap/run{0:d}'.format(run_number),
    #     tlabelfile="../data/medium_cap/raw_medium_cap/run{0:d}_tlabels".format(run_number),
    #     prlabelfile="../data/medium_cap/raw_medium_cap/run{0:d}_prmlabels".format(run_number))
    
    #--- saving tlabels and prmlabels from likelihoods files  ----
    # dummya, dummyb, prs = getlabels("results/run{0:d}_likelihoods".format(run_number), tlabelFile="results/run{0:d}_tlabels".format(run_number), prlabelFile="results/run{0:d}_prmlabels".format(run_number))
    # getlabels("results/run{0:d}_likelihoods_T0".format(run_number), 
    #     tlabelFile="results/run{0:d}_tlabels".format(run_number), 
    #     prlabelFile="results/run{0:d}_prmlabels".format(run_number)) #SEE ME
    
    #--- creating prm files? ---
    # time, X = read_data1('../data/medium_cap/raw_medium_cap/run' + str(run_number), '../data/medium_cap/raw_medium_cap/bias.force',output_fmt='array')
    # N = len(prs)
    # headerstr = "time pos_x pos_y pos_z ori_x ori_y ori_z vel_x vel_y vel_z angvel_x angvel_y angvel_z Fx Fy Fz Mx My Mz Pr"
    # # np.savetxt("../data/medium_cap/auto_labelled/run{0:d}_labelled".format(run_number),np.hstack((np.reshape(time[:N],(N,1)), X[:N], np.reshape(prs,(N,1)))),header=headerstr)
    # np.savetxt("../data/medium_cap/manually_labelled/run{0:d}_labelled".format(run_number),
    #     write_Pr_file(time,X,"../data/medium_cap/raw_medium_cap/run{0:d}_tlabels".format(run_number),
    #         "../data/medium_cap/raw_medium_cap/run{0:d}_prmlabels".format(run_number)),
    #     header=headerstr)

    #--- compute and plot success rate
    # success_rate = compute_success_rate("results/run{0:d}_likelihoods_T0".format(run_number), 
    #     "../data/medium_cap/raw_medium_cap/run{0:d}_tlabels".format(run_number),
    #     "../data/medium_cap/raw_medium_cap/run{0:d}_prmlabels".format(run_number),
    #     failureFile=failFileName) #SEE ME
    # print("run: {0:d} success_rate: {1:f}".format(run_number, success_rate)) #SEE ME
    
    #--- plotting ---  
    # rc('text',usetex=True)
    # rcParams['axes.titlesize'] = 'x-large'
    # rcParams['axes.labelsize'] = 'large'
    # rcParams['xtick.labelsize'] = 'x-large'
    # rcParams['ytick.labelsize'] = 'x-large'
    # plot_file('../data/medium_cap/raw_medium_cap/run{0:d}'.format(run_number),
    #     tlabelfile="results/run{0:d}_tlabels".format(run_number),
    #     prlabelfile="results/run{0:d}_prmlabels".format(run_number))
    # plot_file('../data/medium_cap/raw_medium_cap/run{0:d}'.format(run_number),
    #     tlabelfile="results/run{0:d}_tlabels".format(run_number),
    #     prlabelfile="results/run{0:d}_prmlabels".format(run_number),
    #     tlabelfileTruth="../data/medium_cap/raw_medium_cap/run{0:d}_tlabels".format(run_number),
    #     prlabelfileTruth="../data/medium_cap/raw_medium_cap/run{0:d}_prmlabels".format(run_number),
    #     plot_pos=True,plot_ori=True, plot_vel=True,plot_force=True)
    # plot_file('../data/medium_cap/raw_medium_cap/run1'.format(run_number),
    #     tlabelfile="../data/medium_cap/raw_medium_cap/run1_tlabels".format(run_number),
    #     prlabelfile="../data/medium_cap/raw_medium_cap/run1_prmlabels".format(run_number))
    # plot_file('../data/medium_cap/raw_medium_cap/run2')
    # plt.savefig("figures2label/cap/run2_raw.png",dpi=600, bbox_inches = 'tight',pad_inches = 0)
    # plt.close()