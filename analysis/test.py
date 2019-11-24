from GMM import GMM
n_primitives=5
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
myGMM.initialize_clusters(n_primitives, means0=mu0, cov0=cov0)
# myGMM.initialize_clusters(n_primitives, constraints=myConstraints,means0=mu0, cov0=cov0)
myGMM.initialize_clusters_from_savedfiles(n_primitives, 'references/mean', 'references/covar', 'references/pi')
for i in range(20):
    myGMM.expectation_step()
    myGMM.maximization_step()
