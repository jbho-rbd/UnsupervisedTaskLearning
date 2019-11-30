import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from scipy.spatial.transform import Rotation as R
from matplotlib import rc
import bisect
class Obs():
    def __init__(self, pos, ori, vel, ang_vel, force, moment):
        self.pos = pos
        self.ori = ori
        self.vel = vel
        self.ang_vel = ang_vel
        self.F = force
        self.M = moment
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
def read_data1(runfile,forcebias,t0=0,t1=-1,output_fmt='',tpairlist=None):
    dat=np.genfromtxt(runfile,skip_header=2)
    #collect final poses before truncating
    r_final_inv = R.from_quat(dat[-1,[1,3,4,2]]).inv()
    #final pose
    p1_final = dat[-1,[7,5,6]]
    t=dat[:,0] - dat[0,0]
    FM=dat[0,9:15]
    #now truncate the data
    if tpairlist:
        idxs=np.array([],dtype=int)
        for tpair in tpairlist:
            i0 = bisect.bisect(t,tpair[0])
            i1 = bisect.bisect(t,tpair[1])
            idxs = np.hstack((idxs, np.arange(i0,i1)))
        dat = dat[idxs]
    elif t1 > 0:
        i0 = bisect.bisect(t,t0)
        i1 = bisect.bisect(t,t1)
        dat = dat[i0:i1]
    dat = dat[::25]
    t=dat[:,0] - dat[0,0]
    # FM=np.genfromtxt(forcebias)
    F0=FM[:3]
    M0=FM[3:]
    N = dat.shape[0]
    #pos of rigid bodies
    p1=dat[:,[7,5,6]]-p1_final
    q1=dat[:,[1,3,4,2]]
    #optiforce
    F=dat[:,[9,10,11]]-F0
    M=dat[:,[12,13,14]]-M0
    #need to get poses
    poses = np.zeros((N, 4, 4))
    omegas = np.zeros((N,3))
    euler = np.zeros((N,3))
    vels = np.zeros((N,3))
    ks = np.zeros(3)#counter for the revolutions for x y z
    for i in range(N):
        poses[i,0:3,3] = p1[i] - p1_final
        r = R.from_quat(q1[i])*r_final_inv;
        euler[i] = r.as_euler('zyx', degrees=True) + ks*360
        if(i > 1):
            for j in range(3):
                da = euler[i][j] - euler[i-1][j]
                if (da > 300):
                    ks[j] -= 1
                    euler[i][j] -=360
                if (da < -300):
                    ks[j] += 1
                    euler[i][j] += 360
        poses[i,0:3,0:3] = r.as_dcm()
    # euler -= euler[-1,:]
    for i in range(N):
        if i == 0:
            dR = (poses[1,0:3,0:3] - poses[0,0:3,0:3])/(t[1] - t[0])
            vels[0] = (p1[1] - p1[0])/(t[1] - t[0])
        elif i == N - 1:
            dR = (poses[-1,0:3,0:3] - poses[-2,0:3,0:3])/(t[-1] - t[-2])
            vels[-1] = (p1[-1] - p1[-2])/(t[-1] - t[-2])
        else:
            dR = (poses[i+1,0:3,0:3] - poses[i-1,0:3,0:3])/(t[i+1] - t[i-1])
            vels[i] = (p1[i+1] - p1[i-1])/(t[i+1] - t[i-1])
        skew_angularvelocity = poses[i,0:3,0:3].transpose().dot(dR)
        omegas[i,0] = skew_angularvelocity[2,1]
        omegas[i,1] = skew_angularvelocity[0,2]
        omegas[i,2] = skew_angularvelocity[1,0]
        # k = 5.0
        # omegas[i,2] = k/(k + omegas[i,2]*omegas[i,2])
    if output_fmt == 'obs':#output as list of observations
        obs_list = []
        for i in range(N):
            obs_list.append(Obs(p1[i],q1[i],vels[i], omegas[i], F[i], M[i]))
        return t, obs_list
    elif output_fmt == 'array':
        temp = np.hstack((p1,np.pi/180.0*euler[:,[2,1,0]],vels,omegas,F,M))
        # k = 0.2
        return t, temp
    else:
        return t, p1, vels, euler, omegas, F, M

