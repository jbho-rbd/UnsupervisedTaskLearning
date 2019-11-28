import imageio
import matplotlib.animation as ani
import matplotlib.cm as cmx
import matplotlib.colors as colors
import matplotlib.pyplot as plt
import numpy as np
import scipy as scipy
import enum
import sys
import bisect
import random
import collections
import os

from matplotlib.patches import Ellipse
from PIL import Image
from sklearn import datasets
from sklearn.cluster import KMeans

from classifier import Pr
from read_data import read_data0, read_data1

def sample_primitive(p):
      #p probability distribution of which primitive
      return bisect.bisect(np.cumsum(p), random.random())


# look at the data for each primitive once, run the suggested tests and see which ones yield statistically relevant results, then include those that are statistically relevant in the tree

def initializeTransitionMatrix():
    #transition matrix
    T = np.ones((5,5))
    #
    T[Pr.none.value, Pr.none.value] = 50
    # T[Pr.none.value, Pr.fsm.value] = 0.05
    #
    T[Pr.fsm.value, Pr.fsm.value] = 50
    # T[Pr.fsm.value, Pr.none.value] = 0.05
    # T[Pr.fsm.value, Pr.contact.value] = 0.05
    #
    # T[Pr.contact.value, Pr.fsm.value] = 0.0
    T[Pr.contact.value, Pr.contact.value] = 50
    # T[Pr.contact.value, Pr.screw.value] = 0
    # T[Pr.contact.value, Pr.alignthreads.value] = 0.2
    #
    T[Pr.alignthreads.value, Pr.alignthreads.value] = 50
    # T[Pr.alignthreads.value, Pr.screw.value] = 0.1
    #
    # T[Pr.screw.value, Pr.none.value] = 0.5
    T[Pr.screw.value, Pr.screw.value] = 50
    # T[Pr.screw.value, Pr.contact.value] = 0
    T = np.transpose(T.transpose() / np.sum(T,axis=1))
    return T
def mixWithIdentity(T,alpha):
    return alpha*np.eye(T.shape[0]) + (1 - alpha)*T
def forward_model_primitive(s_value, T):
    #s is a primitve idx
    #T is the transition matrix
    return sample_primitive(T[s_value])
def gaussian(X, mu, cov):
    return scipy.stats.multivariate_normal.pdf(X, mean=mu, cov=cov)
class GMM:
    def __init__(self, X):
        self.X = X;
        self.epoch = 0;
    def initialize_clusters(self, n_clusters, constraints=None, means0=None, cov0=None):#each cluster is a primitive
        self.clusters = []
        self.n_clusters = n_clusters
        idx = np.arange(X.shape[0])
        
        # We could use the KMeans centroids to initialise the GMM
        # Or we can prescribe them
        if means0 is not None:
            if means0.shape[0] != n_clusters or means0.shape[1] != self.X.shape[1]:
                print("means not the correct shape")
                exit()
            mu_k = means0;
        else:
            kmeans = KMeans().fit(X)
            mu_k = kmeans.cluster_centers_
        if constraints is not None:
            self.constraints = True
        else:
            self.constraints = False

        self.likelihoods = np.zeros((X.shape[0], n_clusters))
        for i in range(n_clusters):
            if cov0 is not None:
                cov_k = cov0[i]
            else:
                cov_k = np.identity(self.X.shape[1], dtype=np.float64)
            self.clusters.append({
                'pi_k': 1.0 / n_clusters,
                'mu_k': mu_k[i],
                'cov_k': cov_k
            })
            self.likelihoods[:,i] = 1.0/n_clusters
            if self.constraints:
                self.clusters[i]['constraint_k'] = constraints[i]
        return self.clusters
    def initialize_clusters_from_savedfiles(self, n_clusters, meanfile,covfile,pifile, constraints=None):#each cluster is a primitive
        self.clusters = []
        self.n_clusters = n_clusters
        mu_k = np.load(meanfile)
        cov_k = np.load(covfile)
        pi_k = np.load(pifile)
        if constraints is not None:
            self.constraints = True
        else:
            self.constraints = False
        self.likelihoods = np.zeros((X.shape[0], n_clusters))
        for i in range(n_clusters):
            self.clusters.append({
                'pi_k': pi_k[i],
                'mu_k': mu_k[i],
                'cov_k': 1.5*cov_k[i]
            })
            if self.constraints:
                self.clusters[i]['constraint_k'] = constraints[i]
        return self.clusters
    def expectation_step(self,t=None,prefix="figures/likelihood",saveFile=None,T_matrix=None, T_matrix_standard=None):
        # computes realisations or whatever you wanna call them
        # computes p(belong to primitive | X) for each X
        # this could possibly be replaced with the particle filter
        plotFlag = t is not None
        if plotFlag:
            f,ax = plt.subplots(1)
        if T_matrix is not None:
            self.apf_expectation(T_matrix)
        elif T_matrix_standard is not None:
            self.standard_expectation(T_matrix_standard)
        else:
            self.standard_expectation()
        if plotFlag:
            for kk, cluster in enumerate(self.clusters):
                ax.plot(t,cluster['gamma_nk'],label=Pr(kk))
        if plotFlag:
            ax.legend()
            ax.set_title("Epoch: {0:d}, Likelihood: {1:e}".format(self.epoch, self.get_likelihood()))
            plt.savefig(prefix+"{0:d}.png".format(self.epoch),dpi=600)
            plt.show()
        self.epoch += 1
        if saveFile is not None:
            likelihoods = np.zeros((len(t), self.n_clusters + 1))
            likelihoods[:,0] = t
            for kk, cluster in enumerate(self.clusters):
                likelihoods[:,kk+1] = cluster['gamma_nk']
            np.savetxt(saveFile, likelihoods)
    def standard_expectation(self,T_matrix=None):
        totals = np.zeros(self.X.shape[0], dtype=np.float64)
        if T_matrix is not None:
            for kk, cluster in enumerate(self.clusters):
                self.likelihoods[:,kk] = gaussian(self.X, cluster['mu_k'], cluster['cov_k']).astype(np.float64)
            likelihoods1 = np.zeros((self.X.shape[0],self.n_clusters))#first index time, second index different clusters
            # likelihoods1[1:] = self.likelihoods[:-1].dot(T_matrix)
            # likelihoods1[0,:] = self.likelihoods[0,:]
            likelihoods1[-1,:] = self.likelihoods[-1,:]
            for t in range(self.X.shape[0]-1):
                for s in range(self.n_clusters):
                    for safter in range(self.n_clusters):
                        likelihoods1[t,s] += (1
                            *T_matrix[s,safter]*self.likelihoods[t+1,safter])
                # likelihoods1[t,:] /= np.sum(likelihoods1[t,:])
        for kk, cluster in enumerate(self.clusters):
            if T_matrix is not None:
                gamma_nk = likelihoods1[:,kk]*self.likelihoods[:,kk]
            else:
                gamma_nk = (cluster['pi_k'] * gaussian(self.X, cluster['mu_k'], cluster['cov_k'])).astype(np.float64)
            totals += gamma_nk
            cluster['gamma_nk'] = gamma_nk 
        self.totals = totals
        for kk, cluster in enumerate(self.clusters):
            for i in range(len(totals)):
                if totals[i] == 0.0:
                    cluster['gamma_nk'][i] = 1.0 / self.n_clusters
                    totals[i] = 1e-300
                else:
                    cluster['gamma_nk'][i] /= totals[i];
            self.likelihoods[:,kk] = cluster['gamma_nk']
    def pf_expectation(self,T_forward):
        """ 
        Input:
          observations: states Starting from T=1
          pose_0: (4,4) numpy arrays, starting pose
        Output:
          p_primitives (N x n_primtives probability array)
        """
        N = self.X.shape[0]
        likelihoods = np.zeros((self.X.shape[0], self.n_clusters))
        self.totals = np.zeros(self.X.shape[0])
        for kk, cluster in enumerate(self.clusters):
            likelihoods[0,kk] = (cluster['pi_k'] * gaussian(self.X[0], cluster['mu_k'], cluster['cov_k'])).astype(np.float64)
        self.totals[0] = np.sum(likelihoods[0,:])
        likelihoods[0,:] /= np.sum(likelihoods[0,:])
        N_particles = 100;
        #store primitives as integers
        s = np.zeros((N_particles,N),dtype=int)
        for i in range(N_particles):
          s[i,0] = sample_primitive(likelihoods[0])
        weights = np.ones(N_particles)
        ps = np.zeros(self.n_clusters)
        for t in range(N-1):
          for kk, cluster in enumerate(self.clusters):
            ps[kk] = (cluster['pi_k'] * gaussian(self.X[t+1], cluster['mu_k'], cluster['cov_k'])).astype(np.float64)
          for i in range(N_particles):
            s[i,t+1]=forward_model_primitive(s[i,t])
            weights[i] = ps[s[i,t+1]]
          #normalize weights
          weights = weights / np.sum(weights)
          #resample
          rand_offset = np.random.rand()
          cumweights = np.cumsum(weights)
          averageweight = cumweights[-1]/N_particles
          n_particles_allocated = 0
          for i, cumweight in enumerate(cumweights):
            n = int(np.floor(cumweight / averageweight - rand_offset)) + 1 #n particles that need to be allocated
            # print(n_particles_allocated, n)
            for particle in range(n_particles_allocated, n):
              s[particle,t+1] = s[i,t+1]
            n_particles_allocated = n
          #count primtivies
          temp = collections.Counter(s[:,t+1])
          for kk in range(self.n_clusters):
              likelihoods[t+1, kk] = temp[kk]/N_particles
          self.totals[t+1] = np.sum(ps)
        for kk, cluster in enumerate(self.clusters):
            cluster['gamma_nk'] = likelihoods[:,kk]
    def apf_expectation(self,T_forward):
        """ 
        auxiliary particle filter: https://people.maths.bris.ac.uk/~manpw/apf_chapter.pdf
        Input:
          observations: states Starting from T=1
          pose_0: (4,4) numpy arrays, starting pose
        Output:
          p_primitives (N x n_primtives probability array)
        """
        N = self.X.shape[0]
        likelihoods = np.zeros((self.X.shape[0], self.n_clusters))
        self.totals = np.zeros(self.X.shape[0])
        for kk, cluster in enumerate(self.clusters):
            likelihoods[0,kk] = (cluster['pi_k'] * gaussian(self.X[0], cluster['mu_k'], cluster['cov_k'])).astype(np.float64)
        self.totals[0] = np.sum(likelihoods[0,:])
        likelihoods[0,:] /= np.sum(likelihoods[0,:])
        N_particles = 100;
        #store primitives as integers
        s = np.zeros((N_particles,N),dtype=int)
        for i in range(N_particles):
          s[i,0] = sample_primitive(likelihoods[0])
        weights = np.ones(N_particles)
        alpha = np.zeros(N_particles)
        ps = np.zeros(self.n_clusters)
        p_x1_for_s1 = np.zeros(self.n_clusters)
        for t in range(N-1):
          for s1, cluster1 in enumerate(self.clusters):
            p_x1_for_s1[s1] = (gaussian(self.X[t+1], cluster1['mu_k'], cluster1['cov_k'])).astype(np.float64)
          for s0, cluster0 in enumerate(self.clusters):
            ps[s0] = 0.0
            for s1, cluster1 in enumerate(self.clusters):
                ps[s0] += T_forward[s0,s1]*p_x1_for_s1[s1]
          # print("t: {0:f}, ps".format(t), ps)
          for i in range(N_particles):
            weights[i] = ps[s[i,t]]
          self.totals[t+1] = np.max(weights)
          #normalize weights
          weights = weights / np.sum(weights)
          #resample
          rand_offset = np.random.rand()
          cumweights = np.cumsum(weights)
          averageweight = cumweights[-1]/N_particles
          n_particles_allocated = 0
          for i, cumweight in enumerate(cumweights):
            n = int(np.floor(cumweight / averageweight - rand_offset)) + 1 #n particles that need to be allocated
            for particle in range(n_particles_allocated, n):
              s[particle,t] = s[i,t]
              alpha[particle] = alpha[i]
            n_particles_allocated = n
          #finished resample
          # for s1, cluster1 in enumerate(self.clusters):
          #   ps[s1] = (gaussian(self.X[t+1], cluster1['mu_k'], cluster1['cov_k'])).astype(np.float64)
          for i in range(N_particles):
            s[i,t+1]=forward_model_primitive(s[i,t],mixWithIdentity(T_forward,alpha[i]))
            # weights[i] = ps[s[i,t+1]]*T_forward(s[i,t],s[i,t+1])/
          #count primtivies
          temp = collections.Counter(s[:,t])
          for kk in range(self.n_clusters):
              likelihoods[t, kk] = temp[kk]/N_particles
        self.totals[0] = self.totals[1]
        temp = collections.Counter(s[:,-1])
        for kk, cluster in enumerate(self.clusters):
            likelihoods[-1,kk] = temp[-1]/N_particles
            cluster['gamma_nk'] = likelihoods[:,kk]
    def maximization_step(self):
        N = float(self.X.shape[0])
        
        for kk, cluster in enumerate(self.clusters):
            gamma_nk = cluster['gamma_nk']
            cov_k = np.zeros((self.X.shape[1], self.X.shape[1]))
            
            N_k = np.sum(gamma_nk, axis=0) #sum over all the data
            
            pi_k = N_k / N #weights basd on total sums
            mu_k = np.sum(np.tile(gamma_nk,(self.X.shape[1],1)).transpose() * self.X, axis=0) / N_k #means are a weighted sum based on expectation
            if self.constraints:
                for constraint in cluster['constraint_k']:
                    if constraint[0] > -1: #constraint[0] = -1 is used for inactive constraints
                        mu_k[constraint[0]] = constraint[1]
            for j in range(self.X.shape[0]):
                diff = (self.X[j] - mu_k).reshape(-1, 1)
                cov_k += gamma_nk[j] * np.dot(diff, diff.T)
            if self.constraints:
                for constraint in cluster['constraint_k']:
                    if constraint[0] > -1: #constraint[0] = -1 is used for inactive constraints
                        if constraint[2] > 0: # covar constraint active:
                            scalefactor = constraint[2]/np.sqrt(cov_k[constraint[0],constraint[0]])
                            if scalefactor < 1:
                                cov_k[constraint[0],:] = scalefactor*cov_k[constraint[0],:]
                                cov_k[:,constraint[0]] = scalefactor*cov_k[:,constraint[0]]


            cov_k /= N_k
            # print("Cluster[{0:d}] cond: {1:e}, N_k: {2:e}".format(kk,np.linalg.cond(cov_k), N_k))
            
            cluster['pi_k'] = pi_k
            cluster['mu_k'] = mu_k
            cluster['cov_k'] = cov_k
    def get_likelihood(self):
        sample_likelihoods = np.log(self.totals)
        return np.sum(sample_likelihoods)
    def save(self, meanfile, covarfile, pifile):
        mu0 = np.zeros((self.n_clusters,self.X.shape[1]))
        cov0 = np.zeros((self.n_clusters,self.X.shape[1],self.X.shape[1]))
        pi0 = np.zeros(self.n_clusters)
        for kk, cluster in enumerate(self.clusters):
            mu0[kk] = cluster['mu_k']
            cov0[kk] = cluster['cov_k']
            pi0[kk] = cluster['pi_k']
        np.save(meanfile,mu0)
        np.save(covarfile, cov0)
        np.save(pifile,pi0)
if __name__ == "__main__":
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
        # 'ang_vel_z_transform' : 1
    # subset = np.hstack((np.arange(3, 5), np.arange(6,18)))
    subset = np.hstack((np.arange(6,18)))
    print(subset)
    for key, val in var_idxs.items():
        found_idxs = np.where(subset==val)[0]
        if found_idxs.size > 0:
            var_idxs[key] = found_idxs[0]
        else:
            var_idxs[key] = -1
    n_primitives = 5
    N = len(subset)
    # By manually labelling the data extract some mean and cov data to begin with
    mu0 = np.zeros((n_primitives,N))
    cov0 = np.zeros((n_primitives,N,N))
    tlabels = np.genfromtxt("../data2/run1_tlabels",dtype=float)
    tlabels = np.insert(tlabels,0,0.0)
    labels=[Pr(int(idx)) for idx in np.genfromtxt("../data2/run1_prmlabels")]
    for prim in [Pr.none, Pr.fsm, Pr.contact, Pr.alignthreads, Pr.screw]:
        tpairs = []
        for i in range(len(labels)):#collect different labels and time periods corresponding to this primitive
            if(labels[i] == prim):
                tpairs.append([tlabels[i],tlabels[i+1]])
        print("Primitive: {0:s}".format(Pr(prim)))
        print(tpairs)
        time, X = read_data1('../data2/run1', '../data2/bias.force',output_fmt='array',tpairlist=tpairs)
        #each row of X is an observation
        #each column of X is a variable
        mu0[prim.value] = np.mean(X[:,subset],axis=0)
        # print(mu0[prim.value])
        cov0[prim.value] = np.cov(X[:,subset],rowvar=False)
        # cov0[prim.value] = np.diag(np.diag(cov0[prim.value]))
        # print(np.linalg.cond(cov0[prim.value]))
        # print("{0:e}".format(np.linalg.cond(cov0[prim.value])))
    #TRAINING
    time,X = read_data1('../data2/run1', '../data2/bias.force',output_fmt='array',t0=0.0, t1 = 10.5)
    # np.savetxt("scaling_cov_diag.dat",np.diag(np.cov(X,rowvar=False)))
    #set up my Constraints
    myConstraints=[()]*n_primitives
    myConstraints[Pr.none.value] = (
        (var_idxs['vel_x'], 0.0, -1.0),
        (var_idxs['vel_y'], 0.0, -1.0),
        (var_idxs['vel_z'], 0.0, -1.0),
        (var_idxs['ang_vel_x'], 0.0, -1.0),
        (var_idxs['ang_vel_y'], 0.0, -1.0),
        (var_idxs['ang_vel_z'], 0.0, -1.0),
        (var_idxs['F_x'], 0.0, -1.0),
        (var_idxs['F_y'], 0.0, -1.0),
        (var_idxs['F_z'], 0.0, -1.0),
        (var_idxs['M_x'], 0.0, -1.0),
        (var_idxs['M_y'], 0.0, -1.0),
        (var_idxs['M_z'], 0.0, -1.0),
        )
    myConstraints[Pr.fsm.value] = (
        (var_idxs['M_x'], 0.0, -1.0),
        (var_idxs['M_y'], 0.0, -1.0),
        (var_idxs['M_z'], 0.0, -1.0)
    )
    myConstraints[Pr.contact.value] = (
        (var_idxs['vel_x'], 0.0, -1.0),
        (var_idxs['vel_y'], 0.0, -1.0),
        (var_idxs['vel_z'], 0.0, -1.0),
        (var_idxs['ang_vel_z'], 0.0, 0.5),
    )
    myConstraints[Pr.screw.value] = (
        (var_idxs['ori_x'], 0.0, -1.0),
        (var_idxs['ori_y'], 0.0, -1.0),
        (var_idxs['vel_x'], 0.0, -1.0),
        (var_idxs['vel_y'], 0.0, -1.0),
        (var_idxs['vel_z'], 0.0, -1.0),
        (var_idxs['ang_vel_x'], 0.0, -1.0),
        (var_idxs['ang_vel_y'], 0.0, -1.0)
    )
    myConstraints[Pr.alignthreads.value] = (
        (var_idxs['vel_x'], 0.0, -1.0),
        (var_idxs['vel_y'], 0.0, -1.0),
        (var_idxs['vel_z'], 0.0, -1.0),
        (var_idxs['ang_vel_x'], 0.0, -1.0),
        (var_idxs['ang_vel_y'], 0.0, -1.0),
        (var_idxs['ang_vel_z'], 0.0, 0.5)
    )
    myGMM = GMM(X[:,subset])
    # myGMM.initialize_clusters(n_primitives, means0=mu0, cov0=cov0)
    transition = initializeTransitionMatrix()
    if False:
        myGMM.initialize_clusters(n_primitives, means0=mu0, cov0=cov0,
            constraints=myConstraints)
        for i in range(30):
            if i == 29:#i % 100 == 0:
                myGMM.expectation_step(t=time,saveFile="results/run1_likelihoods"
                    ,T_matrix=transition
                    )
            else:
                myGMM.expectation_step(
                    T_matrix=transition
                    )
            myGMM.maximization_step()
            print("it: {0:d} likelihood function {1:e}".format(i, myGMM.get_likelihood()))
        myGMM.save('references/mean', 'references/covar', 'references/pi')
        means = np.load('references/mean.npy')
        covar = np.load('references/covar.npy')
        for prim in [Pr.none, Pr.fsm, Pr.contact, Pr.alignthreads, Pr.screw]:
            print("Means: ", prim)
            for var in ('ang_vel_z', 'F_z'):
                print("{0:s}: mu: {1:f} stdev ".format(var,means[prim.value][var_idxs[var]]),
                    np.sqrt(covar[prim.value,var_idxs[var],var_idxs[var]])) 
    os.system('python3 plot_data.py 1')
    #TESTING
    if len(sys.argv) < 2:
        exit()
    run_number=int(sys.argv[1])
    testfile='../data2/run{0:d}'.format(run_number)
    print("--------testing: ",testfile, "-----------")
    time,X = read_data1(testfile, '../data2/bias.force',output_fmt='array')
    mytestGMM = GMM(X[:,subset])
    mytestGMM.initialize_clusters_from_savedfiles(n_primitives, 'references/mean.npy', 'references/covar.npy', 'references/pi.npy',constraints=myConstraints)
    l = -1e30
    for i in range(30):
        if i == 29:
            mytestGMM.expectation_step(t=time,prefix="figures/run{0:d}_epoch".format(run_number),saveFile="results/run{0:d}_likelihoods".format(run_number),
                T_matrix=transition)
        else:
            mytestGMM.expectation_step(
                T_matrix=transition)
        mytestGMM.maximization_step()
        # l_old = l
        # l = mytestGMM.get_likelihood()
        # if abs(l/l_old - 1) < 1e-14 :
        #     break;
        print("it: {0:d} likelihood  function {1:e}".format(i, mytestGMM.get_likelihood()))
    mytestGMM.save('references/meantest', 'references/covartest', 'references/pitest')
    means = np.load('references/meantest.npy')
    covar = np.load('references/covartest.npy')
    for prim in [Pr.none, Pr.fsm, Pr.contact, Pr.alignthreads, Pr.screw]:
        print("Stats: ", prim)
        for var in ('ang_vel_z', 'F_z'):
            print("{0:s}: mu: {1:f} stdev ".format(var,means[prim.value][var_idxs[var]]),
            np.sqrt(covar[prim.value,var_idxs[var],var_idxs[var]])) 
    os.system('python3 plot_data.py '+str(run_number))


