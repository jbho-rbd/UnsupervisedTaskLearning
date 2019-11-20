import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from scipy.spatial.transform import Rotation as R
from matplotlib import rc
from read_data import read_data0, read_data1
from classifier import Pr

#y axis becomes z
#z axis becomes x
#x axis becomes y
# t, p1, euler, omegas, F, M = read_data0('../data/run1.dat', '../data/bias.force')
t, p1, vels, euler, omegas, F, M = read_data1('../data2/run1', '../data2/bias.force')#,t0=0,t1=2.0)
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
ax[2].plot(t,euler[:,2],'r',label='$\\theta$')
ax[2].plot(t,euler[:,1],'g',label='$\Phi$')
ax[2].plot(t,euler[:,0],'b',label='$\Psi$')
ax[2].set_ylabel('degrees')
ax[2].legend(loc='upper right')
ax[3].plot(t,omegas[:,0]*180/np.pi,'r',label='$\omega_x$')
ax[3].plot(t,omegas[:,1]*180/np.pi,'g',label='$\omega_y$')
ax[3].plot(t,omegas[:,2]*180/np.pi,'b',label='$\omega_z$')
ax[3].set_ylabel('deg/s')
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
vlines = np.genfromtxt("../data2/run1_tlabels",dtype=float)
vlines = np.insert(vlines,0,0.0)
# vlines=np.array([0.0, 1.08, 3.27,4.2,5.5,6.72,6.9,7.44, 7.6, 8.02, 8.21, 8.68, 8.98, 10.5])
# np.savetxt("../data2/run1_tlabels",vlines[1:])
print(np.genfromtxt("../data2/run1_prmlabels"))
labels=[Pr(int(idx)) for idx in np.genfromtxt("../data2/run1_prmlabels")]
# np.savetxt("../data2/prm_tlabels",[label.value for label in labels])
for i in range(7):
	for vline in vlines[1:-1]:
		ax[i].axvline(x=vline,color='k',linestyle=':')
y = 0.5
xcords = 0.5*(vlines[1:] + vlines[:-1])
for i in range(len(xcords)):
	ax[6].text(xcords[i],y,labels[i],horizontalalignment='center',rotation=90, verticalalignment='center')
# ax[6].text(xcords[1],y,'Contact',horizontalalignment='center')
# ax[6].text(xcords[2],y,'Align',horizontalalignment='center')
# ax[6].text(xcords[3],y,'Screw',horizontalalignment='center')
# ax[6].text(xcords[4],y,'Tighten',horizontalalignment='center')
ax[6].set_xlabel('time')
ax[6].get_yaxis().set_ticks([])
labels=('A','B','C','D','E','F')
for i in range(6):
	secaxy = ax[i].twinx()
	secaxy.set_ylabel(labels[i],rotation=0,labelpad=10)
	secaxy.get_yaxis().set_ticks([])
f.align_ylabels(ax)
plt.show()
