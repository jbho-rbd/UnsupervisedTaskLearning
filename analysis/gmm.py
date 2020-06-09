"""======================================================================================
 gmm.py
 
 Input: raw sensor data    
 Output: labelled sensor data
 
Jonathan Ho, Fall 2019
Elena Galbally, Spring 2020
======================================================================================"""
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

from read_data import read_data0, read_data1

""" --------------------------------------------------------------------------------------
   Utility Functions
-----------------------------------------------------------------------------------------"""
class Pr(enum.Enum): 
    """ 
        Enum associating each primitive with an integer 
    """
    none = 0
    fsm = 1
    align = 2
    engage = 3
    screw = 4
    tighten = 5

def sample_primitive(p):
    """ 
        Selects the primitive with the highest associated probability
            Input:
                p: (6,) numpy array representing the probability of each primitive
            Output:
                integer between 0-5 corresponding to the selected primitive(2,)
    """
    return bisect.bisect(np.cumsum(p), random.random())

def initializeTransitionMatrix(final=False):
    """ 
            Input:
                final: flag -> False for Training, True for Testing
            Output:
                array of size (6,6) containing conditional probabilities: T[pr_i|pr_j]
    """
    if final:
        T = np.zeros((6,6)) # you can only move between specified primitives (more restrictive)
    else:
        T = np.ones((6,6)) # you can move from any primitive to any other
    # 
    T[Pr.none.value, Pr.none.value] = 50
    T[Pr.none.value, Pr.fsm.value] = 1
    T[Pr.none.value, Pr.screw.value] = 1
    # T[Pr.none.value, Pr.fsm.value] = 0.05
    #
    T[Pr.fsm.value, Pr.fsm.value] = 50
    T[Pr.fsm.value, Pr.align.value] = 1
    # T[Pr.fsm.value, Pr.none.value] = 0.05
    #
    # T[Pr.align.value, Pr.fsm.value] = 0.0
    T[Pr.align.value, Pr.align.value] = 50
    # T[Pr.align.value, Pr.screw.value] = 0
    T[Pr.align.value, Pr.engage.value] = 1
    #
    T[Pr.engage.value, Pr.engage.value] = 50
    T[Pr.engage.value, Pr.none.value] = 1
    T[Pr.engage.value, Pr.screw.value] = 1
    # T[Pr.engage.value, Pr.tighten.value] = 0
    # T[Pr.engage.value, Pr.screw.value] = 0.1
    
    # T[Pr.screw.value, Pr.none.value] = 0.5
    T[Pr.screw.value, Pr.screw.value] = 50
    T[Pr.screw.value, Pr.none.value] = 1
    T[Pr.screw.value, Pr.tighten.value] = 1
    # T[Pr.screw.value, Pr.engage.value] = 0
    # T[Pr.screw.value, Pr.align.value] = 0

    T[Pr.tighten.value, Pr.tighten.value] = 50
    T[Pr.tighten.value, Pr.none.value] = 1
    # T[Pr.tighten.value, Pr.screw.value] = 0

    # scale values so they are all probabilities between 0-1
    T = np.transpose(T.transpose() / np.sum(T,axis=1))
    return T

def initializeConstraints():
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
    myConstraints[Pr.align.value] = (
        (var_idxs['vel_x'], 0.0, -1.0),
        (var_idxs['vel_y'], 0.0, -1.0),
        (var_idxs['vel_z'], 0.0, -1.0),
        (var_idxs['ang_vel_z'], 0.0, 0.5),
    )
    myConstraints[Pr.engage.value] = (
        (var_idxs['ori_x'], 0.0, 0.5),
        (var_idxs['ori_y'], 0.0, 0.5),
        (var_idxs['vel_x'], 0.0, -1.0),
        (var_idxs['vel_y'], 0.0, -1.0),
        (var_idxs['vel_z'], 0.0, -1.0),
        (var_idxs['ang_vel_x'], 0.0, -1.0),
        (var_idxs['ang_vel_y'], 0.0, -1.0)
    )
    myConstraints[Pr.screw.value] = (
        (var_idxs['vel_x'], 0.0, -1.0),
        (var_idxs['vel_y'], 0.0, -1.0),
        (var_idxs['vel_z'], 0.0, -1.0),
        (var_idxs['ang_vel_x'], 0.0, -1.0),
        (var_idxs['ang_vel_y'], 0.0, -1.0),
    )
    myConstraints[Pr.tighten.value] = (
        (var_idxs['vel_x'], 0.0, -1.0),
        (var_idxs['vel_y'], 0.0, -1.0),
        (var_idxs['vel_z'], 0.0, -1.0),
        (var_idxs['ang_vel_x'], 0.0, -1.0),
        (var_idxs['ang_vel_y'], 0.0, -1.0),
        (var_idxs['ang_vel_z'], 0.0, 0.5)
    )
    return myConstraints

def mixWithIdentity(T,alpha):
    return alpha*np.eye(T.shape[0]) + (1 - alpha)*T

def forward_model_primitive(s_value, T):
    #s is a primitve idx
    #T is the transition matrix
    return sample_primitive(T[s_value])

def gaussian(X, mu, cov):
    return scipy.stats.multivariate_normal.pdf(X, mean=mu, cov=cov)

def mix_mean_covar_pi(mean,covar,pi,mean0,covar0,pi0,k):
    np.save(mean, np.load(mean)*(1 - k) + k*np.load(mean0))
    np.save(covar, np.load(covar)*(1 - k) + k*np.load(covar0))
    np.save(pi, np.load(pi)*(1 - k) + k*np.load(pi0))

""" --------------------------------------------------------------------------------------
   Gaussian Mixture Model Class 
-----------------------------------------------------------------------------------------"""
class GMM:
    def __init__(self, X, offset=0.0):
        self.X = X;
        self.epoch = 0;
        self.offset = offset
    
    def initialize_clusters(self, n_clusters, constraints=None, means0=None, cov0=None):
        """
            Each cluster is a primitive
        """
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
    
    def initialize_clusters_from_savedfiles(self, n_clusters, meanfile,covfile,pifile, constraints=None):
        """
            Each cluster is a primitive
        """
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
                'cov_k': 2.0*cov_k[i]
            })
            if self.constraints:
                self.clusters[i]['constraint_k'] = constraints[i]
        return self.clusters

    def inflate_cov(self, factor):#each cluster is a primitive
        for cluster in self.clusters:
            cluster['cov_k'] = factor*cluster['cov_k']

    def expectation_step(self,t=None,prefix="figures/likelihood",saveFile=None,T_matrix_APF=None, T_matrix_standard=None):
        """
            Output: computes p(belong to primitive | X) for each X
            It saves these likelihoods to a .txt
            Uses: a particle filter with a heuristic forward model: T(sn | sn-1) 
        """
        plotFlag = t is not None
        if plotFlag:
            f,ax = plt.subplots(1)
        if T_matrix_APF is not None:
            self.apf_expectation(T_matrix_APF)  #the option that worked best
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
            # plt.show()
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
        likelihoods[0,0] = 1e10
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
                ps[s0] += T_forward[s0,s1]*(p_x1_for_s1[s1] + self.offset)
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
        self.offset = max(self.offset*0.5, 0.1)

    def maximization_step(self):
        """
            Gaussian: p ( X| s , μ , Σ ) 
            Optimize μ , Σ  max likelihood
            Heuristic constraints on μ , Σ
        """
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
        """
        Save binaries
        """
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

    def manual_labelling(self):
        """---------------------
            Manual Labelling
        ------------------------"""
        # By manually labelling 1 run of data we extract a mean and cov to begin the iterations
        print("-------- manual labelling of run1 -----------")
        mu0 = np.zeros((n_primitives,N))
        cov0 = np.zeros((n_primitives,N,N))
        tlabels = np.genfromtxt("../data/medium_cap/raw_medium_cap/run1_tlabels",dtype=float)
        tlabels = np.insert(tlabels,0,0.0)
        labels=[Pr(int(idx)) for idx in np.genfromtxt("../data/medium_cap/raw_medium_cap/run1_prmlabels")]
        for prim in [Pr.none, Pr.fsm, Pr.align, Pr.engage, Pr.screw, Pr.tighten]:
            tpairs = []
            for i in range(len(labels)):#collect different labels and time periods corresponding to this primitive
                if(labels[i] == prim):
                    tpairs.append([tlabels[i],tlabels[i+1]])
            print("Primitive: {0:s}".format(Pr(prim)))
            print(tpairs)
            time, X = read_data1('../data/medium_cap/raw_medium_cap/run1', 
                '../data/medium_cap/raw_medium_cap/bias.force',output_fmt='array',tpairlist=tpairs)
            #each row of X is an observation
            #each column of X is a variable
            mu0[prim.value] = np.mean(X[:,subset],axis=0)
            cov0[prim.value] = np.cov(X[:,subset],rowvar=False)
        return mu0,cov0

    def train(self, mu0, cov0, numIterTrain, transition):
        """---------------------
            Training
        ------------------------"""      
        myGMM.initialize_clusters(n_primitives, means0=mu0, cov0=cov0, constraints=myConstraints)

        # Train by running gmm for "run1" of the demonstration data    
        print("-------- training on run1 -----------")
        for i in range(numIterTrain):
            if i == numIterTrain - 1: # save data on the last iteration
                myGMM.expectation_step(t=time,saveFile="results/run1_likelihoods",T_matrix_APF=transition)
            else: # T_matrix_APF implies that the expectation step is using an Augmented Particle Filter
                myGMM.expectation_step(T_matrix_APF=transition)
            myGMM.maximization_step()
            print("it: {0:d} likelihood function {1:e}".format(i, myGMM.get_likelihood()))
        
        # Save training data
        myGMM.save('references/mean', 'references/covar', 'references/pi')

        # Print training results
        means = np.load('references/mean.npy')
        covar = np.load('references/covar.npy')
        print("-------- training results for run1 -----------")
        for prim in [Pr.none, Pr.fsm, Pr.align, Pr.engage, Pr.screw, Pr.tighten]:
            print("Means: ", prim)
            for var in ('ang_vel_z', 'F_z'):
                print("{0:s}: mu: {1:f} stdev ".format(var,means[prim.value][var_idxs[var]]),
                    np.sqrt(covar[prim.value,var_idxs[var],var_idxs[var]])) 

        # Plot by running the "main" in plot_data.py
        os.system('python3 plot_data.py 1')
    
    def test(self, run_number, numIterTest, transition):
        """---------------------
        Testing
        ------------------------"""
        """ 
            The following code will run iff you specify a run number on command line:
                python gmm.py [run_number]

            *note: a run is the raw sensor data corresponding to 
            one human demonstration of the full task
        """
        offset = 0.01
        success = False

        # Testing
        print("-------- testing on: ",testfile, "-----------")
        while not success and offset < 10000:
            success = True
            offset = offset*10
            print("offset: ", offset)
            mytestGMM.offset = offset
            try:
                mytestGMM.initialize_clusters_from_savedfiles(n_primitives, 
                    'references/mean.npy', 'references/covar.npy', 'references/pi.npy',constraints=myConstraints)
                for i in range(numIterTest):
                    if i == numIterTest - 1:
                        mytestGMM.expectation_step(t=time,prefix="figures/run{0:d}_epoch".format(run_number),
                            saveFile="results/run{0:d}_likelihoods".format(run_number),
                            T_matrix_APF=transition)
                    else:
                        mytestGMM.expectation_step(T_matrix_APF=transition)
                    
                    mytestGMM.maximization_step()
                    print("it: {0:d} likelihood  function {1:e}".format(i, mytestGMM.get_likelihood()))
            
            except Exception as e:
                print("error: ", e)
                success = False

        # Save testing data
        mytestGMM.save('references/meantest', 'references/covartest', 'references/pitest')
        # mix_mean_covar_pi('references/mean.npy', 'references/covar.npy', 'references/pi.npy',
        #     'references/meantest.npy', 'references/covartest.npy', 'references/pitest.npy',
        #     1.0/run_number)

        # Print testing results
        means = np.load('references/meantest.npy')
        covar = np.load('references/covartest.npy')
        print("-------- testing results for: ",testfile, "-----------")
        for prim in [Pr.none, Pr.fsm, Pr.align, Pr.engage, Pr.screw, Pr.tighten]:
            print("Stats: ", prim)
            for var in ('ang_vel_z', 'M_z'):
                print("{0:s}: mu: {1:f} stdev ".format(var,means[prim.value][var_idxs[var]]),
                np.sqrt(covar[prim.value,var_idxs[var],var_idxs[var]])) 

        # Plot by running the "main" in plot_data.py
        os.system('python3 plot_data.py '+str(run_number))


""" --------------------------------------------------------------------------------------
   Main
-----------------------------------------------------------------------------------------"""
""" 
    - Initialize:
            * create an array "subset" with the raw sensor data of interest
            * manually label one run (data from one human demo of the whole task) and extract 
              a mean and cov based on the manual labelling to seed the gmm algorithm
    - Training:
            * train by running the gmm on run1 using the mean and cov from the manual labelling as seeds
    - Testing:
            * test on other runs using the mean and cov from the training as seeds 
"""
if __name__ == "__main__":

    # Initialize parameters
    n_primitives = 6  
    numIterTrain = 5
    numIterTest = 5  

    # Dictionary for the raw sensor data
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

    # subset contains only the sensor data we are interested in: 3,4,6-17
    subset = np.hstack((np.arange(var_idxs['vel_x'], 5), np.arange(6,18)))
    
    # Reorder the dictionary according to data in subset
    for key, val in var_idxs.items():
        found_idxs = np.where(subset==val)[0]
        if found_idxs.size > 0:
            var_idxs[key] = found_idxs[0]
        else:
            var_idxs[key] = -1 # assign -1 in dictionary to the data that wasn't included in subset

    N = len(subset)

    """
    TRAINING AND TESTING
        labelling and training will run if and only if you don't pass any run numbers
        otherwise it will just test
    """
    trainingFlag = len(sys.argv) < 2 
    myConstraints = initializeConstraints()

    if trainingFlag:  
        # Train data
        time,X = read_data1('../data/medium_cap/raw_medium_cap/run1', 
            '../data/medium_cap/raw_medium_cap/bias.force',
            output_fmt='array',t0=0.0, t1 = 10.5)
        
        # Init
        myGMM = GMM(X[:,subset])
        transition = initializeTransitionMatrix()

        print(">>>>>>>> TRAINING >>>>>>>>")
        mu0,cov0 = myGMM.manual_labelling()
        myGMM.train(mu0, cov0, numIterTrain, transition)
   
    else: 
        # Test data
        run_number=int(sys.argv[1])
        testfile='../data/medium_cap/raw_medium_cap/run{0:d}'.format(run_number)
        time,X = read_data1(testfile, '../data/medium_cap/raw_medium_cap/bias.force',output_fmt='array')
        
        # Init
        mytestGMM = GMM(X[:,subset])
        transition = initializeTransitionMatrix(final=True)

        print(">>>>>>>> TESTING >>>>>>>>")  
        mytestGMM.test(run_number, numIterTest, transition)


    """---------------------
        Update T Matrix
    ------------------------"""
    """ 
        - Update Transition Matrix based on all labelled runs
        - Train and Test again
        - Run this for several iterations until improved labelling 
          success wrt manually labelled runs
    """
    