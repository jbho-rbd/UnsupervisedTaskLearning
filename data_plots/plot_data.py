import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from scipy.spatial.transform import Rotation as R
from matplotlib import rc
# rc('text',usetex=True) 
dat=np.genfromtxt('../data/run1.dat',skip_header=2)
FM=np.genfromtxt('../data/bias.force')
F0=FM[:3]
M0=FM[3:]
N = dat.shape[0]
t=dat[:,0] - dat[0,0]
#quaternions of rigid bodies
q1=dat[:,1:5]
q2=dat[:,5:9]
q3=dat[:,9:13]
#final pose
p1_final = dat[-1,13:16]
r_final_inv = R.from_quat(q1[-1]).inv()
#pos of rigid bodies
p1=dat[:,13:16]-p1_final
p2=dat[:,16:19]
p3=dat[:,19:22]
#timestampoptitrack
t_opti=dat[:,22]
#optiforce
F=dat[:,23:26] - F0
M=dat[:,26:29] - M0
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
	euler[i] = r.as_euler('yzx', degrees=True)
	poses[i,0:3,0:3] = r.as_dcm()
for i in range(N-1):
	dR = (poses[i+1,0:3,0:3] - poses[i,0:3,0:3])/(t[i+1] - t[i])
	skew_angularvelocity = poses[i,0:3,0:3].transpose().dot(dR)
	omegas[i,0] = skew_angularvelocity[2,1]
	omegas[i,1] = skew_angularvelocity[0,2]
	omegas[i,2] = skew_angularvelocity[1,0]
omegas[-1] = omegas[-2]
# fig = plt.figure()
# ax = fig.add_subplot(111, projection='3d')
# quiverlength = 0.03
# poses_sampled = poses[0::50]
# ax.quiver(poses_sampled[:,0,3], poses_sampled[:,1,3], poses_sampled[:,2,3],
#           poses_sampled[:,0,0], poses_sampled[:,1,0], poses_sampled[:,2,0],
#           color='b', length=quiverlength, arrow_length_ratio=0.05)
# ax.quiver(poses_sampled[:,0,3], poses_sampled[:,1,3], poses_sampled[:,2,3],
#           poses_sampled[:,0,1], poses_sampled[:,1,1], poses_sampled[:,2,1],
#           color='g', length=quiverlength, arrow_length_ratio=0.05)
# ax.quiver(poses_sampled[:,0,3], poses_sampled[:,1,3], poses_sampled[:,2,3],
#           poses_sampled[:,0,2], poses_sampled[:,1,2], poses_sampled[:,2,2],
#           color='r', length=quiverlength, arrow_length_ratio=0.05)
# ax.set_xlabel('x')
# ax.set_ylabel('y')
# ax.set_zlabel('z')
f,ax=plt.subplots(5,1,sharex=True,figsize=(8,9))
ax[0].plot(t,p1[:,0],label='pos x')
ax[0].plot(t,p1[:,1],label='pos y')
ax[0].plot(t,p1[:,2],label='pos z')
ax[0].set_ylabel('m')
ax[0].legend()
ax[1].plot(t,euler[:,0],label='Euler y')
ax[1].plot(t,euler[:,1],label='Euler z')
ax[1].plot(t,euler[:,2],label='Euler x')
ax[1].set_ylabel('degrees')
ax[1].legend()
ax[2].plot(t,omegas[:,0],label='$\omega_x$')
ax[2].plot(t,omegas[:,1],label='$\omega_y$')
ax[2].plot(t,omegas[:,2],label='$\omega_z$')
ax[2].set_ylabel('rad/s')
ax[2].legend()
ax[3].plot(t,F[:,0],label='$F_x$')
ax[3].plot(t,F[:,1],label='$F_y$')
ax[3].plot(t,F[:,2],label='$F_z$')
ax[3].set_ylabel('N')
ax[3].legend()
ax[4].plot(t,M[:,0],label='$M_x$')
ax[4].plot(t,M[:,1],label='$M_y$')
ax[4].plot(t,M[:,2],label='$M_z$')
ax[4].set_ylabel('Nm')
ax[4].legend()
plt.xlabel('t')

plt.legend()

plt.show()
