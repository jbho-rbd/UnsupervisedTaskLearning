from gmm import GMM
from read_data import read_data1
from classifier import Pr
import numpy as np
var_idxs = { 
    'pos_x' : 0,
    'pos_y' : 1,
    'pos_z' : 2,
    'ori_x' : 3,
    'ori_y' : 4,
    'ori_z' : 5,
    'vel_x' : 6,
    'vel_y' : 7,
    'vel_z' : 8,
    'ang_vel_x' : 9,
    'ang_vel_y' : 10,
    'ang_vel_z' : 11,
    'F_x' : 12,
    'F_y' : 13,
    'F_z' : 14,
    'M_x' : 15,
    'M_y' : 16,
    'M_z' : 17}
subset = np.arange(6, 18)
for key, val in var_idxs.items():
    found_idxs = np.where(subset==val)[0]
    if found_idxs.size > 0:
        var_idxs[key] = found_idxs[0]
    else:
        var_idxs[key] = -1
n_primitives=5
scaling = 1/np.sqrt(np.genfromtxt("scaling_cov_diag.dat"))
time,X = read_data1('../data2/run2', '../data2/bias.force',output_fmt='array',t0=0.0, t1 = 10.5,scale=scaling)
# np.savetxt("scaling_cov_diag.dat",np.diag(np.cov(X,rowvar=False)))
#set up my Constraints
myConstraints=[()]*n_primitives
myConstraints[Pr.none.value] = (
    (var_idxs['vel_x'], 0.0),
    (var_idxs['vel_y'], 0.0),
    (var_idxs['vel_z'], 0.0),
    (var_idxs['ang_vel_x'], 0.0),
    (var_idxs['ang_vel_y'], 0.0),
    (var_idxs['ang_vel_z'], 0.0),
    (var_idxs['F_x'], 0.0),
    (var_idxs['F_y'], 0.0),
    (var_idxs['F_z'], 0.0),
    (var_idxs['M_x'], 0.0),
    (var_idxs['M_y'], 0.0),
    (var_idxs['M_z'], 0.0),
    )
myConstraints[Pr.fsm.value] = (
    (var_idxs['M_x'], 0.0),
    (var_idxs['M_y'], 0.0),
    (var_idxs['M_z'], 0.0)
)
myConstraints[Pr.contact.value] = (
    (var_idxs['vel_x'], 0.0),
    (var_idxs['vel_y'], 0.0),
    (var_idxs['vel_z'], 0.0)
)
myConstraints[Pr.screw.value] = (
    (var_idxs['vel_x'], 0.0),
    (var_idxs['vel_y'], 0.0),
    (var_idxs['vel_z'], 0.0),
    (var_idxs['ang_vel_x'], 0.0),
    (var_idxs['ang_vel_y'], 0.0)
)
myConstraints[Pr.align.value] = (
    (var_idxs['vel_x'], 0.0),
    (var_idxs['vel_y'], 0.0),
    (var_idxs['vel_z'], 0.0),
    (var_idxs['ang_vel_x'], 0.0),
    (var_idxs['ang_vel_y'], 0.0),
    (var_idxs['ang_vel_z'], 0.0)
)
myGMM = GMM(X[:,subset])
myGMM.initialize_clusters_from_savedfiles(n_primitives, 'references/mean.npy', 'references/covar.npy', 'references/pi.npy')
for i in range(20):
    myGMM.expectation_step()
    myGMM.maximization_step()
