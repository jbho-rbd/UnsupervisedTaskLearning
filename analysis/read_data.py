import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from scipy.spatial.transform import Rotation as R
from matplotlib import rc
#y axis becomes z
#z axis becomes x
#x axis becomes y
def read_data0(runfile,forcebias):
    dat=np.genfromtxt(runfile,skip_header=2)
    FM=np.genfromtxt(forcebias)
    F0=FM[:3]
    M0=FM[3:]
    N = dat.shape[0]
    t=dat[:,0] - dat[0,0]
    #quaternions of rigid bodies
    q1=dat[:,[1,3,4,2]]
    #q2=dat[:,[5,8,6,7]]
    #q3=dat[:,[9,12,10,11]]
    #final pose
    p1_final = dat[-1,[15,13,14]]
    r_final_inv = R.from_quat(q1[-1]).inv()
    #pos of rigid bodies
    p1=dat[:,[15,13,14]]-p1_final
    p2=dat[:,[18,16,17]]
    p3=dat[:,[21,19,20]]
    #timestampoptitrack
    t_opti=dat[:,22]
    #optiforce
    F=dat[:,[23,24,25]] - F0
    M=dat[:,[26,27,28]] - M0
    # plt.figure(0)
    # plt.plot(t,F)
    # plt.figure(1)
    # plt.plot(t,np.linalg.norm(q1,axis=1))
    #need to get poses
    poses = np.zeros((N, 4, 4))
    omegas = np.zeros((N,3))
    euler = np.zeros((N,3))
    for i in range(N):
        poses[i,0:3,3] = p1[i] - p1_final
        r = R.from_quat(q1[i])*r_final_inv;
        euler[i] = r.as_euler('zyx', degrees=True)
        poses[i,0:3,0:3] = r.as_dcm()
    for i in range(N-1):
        dR = (poses[i+1,0:3,0:3] - poses[i,0:3,0:3])/(t[i+1] - t[i])
        skew_angularvelocity = poses[i,0:3,0:3].transpose().dot(dR)
        omegas[i,0] = skew_angularvelocity[2,1]
        omegas[i,1] = skew_angularvelocity[0,2]
        omegas[i,2] = skew_angularvelocity[1,0]
    omegas[-1] = omegas[-2]
    return t, p1, euler, omegas, F, M

