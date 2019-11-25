import imageio
import matplotlib.animation as ani
import matplotlib.cm as cmx
import matplotlib.colors as colors
import matplotlib.pyplot as plt
import numpy as np
import scipy as scipy
import enum

from matplotlib.patches import Ellipse
from PIL import Image
from sklearn import datasets
from sklearn.cluster import KMeans

from classifier import Pr
from read_data import read_data0, read_data1


def gaussian(X, mu, cov):
    return scipy.stats.multivariate_normal.pdf(X, mean=mu, cov=cov)
class GMM:
    def __init__(self, X):
        self.X = X;
        self.epoch = 0;
    def initialize_clusters(self, n_clusters, constraints=None, means0=None, cov0=None):#each cluster is a primitive
        self.clusters = []
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
            if self.constraints:
                self.clusters[i]['constraint_k'] = constraints[i]
        return self.clusters
    def initialize_clusters_from_savedfiles(self, n_clusters, meanfile,covfile,pifile, constraints=None):#each cluster is a primitive
        self.clusters = []
        mu_k = np.load(meanfile)
        cov_k = np.load(covfile)
        pi_k = np.load(pifile)
        if constraints is not None:
            self.constraints = True
        else:
            self.constraints = False

        for i in range(n_clusters):
            self.clusters.append({
                'pi_k': pi_k[i],
                'mu_k': mu_k[i],
                'cov_k': 10*cov_k[i]
            })
            if self.constraints:
                self.clusters[i]['constraint_k'] = constraints[i]
        return self.clusters
    def expectation_step(self,t=None,prefix="figures/likelihood",saveFile=None):
        # computes realisations or whatever you wanna call them
        # computes p(belong to primitive | X) for each X
        # this could possibly be replaced with the particle filter
        totals = np.zeros(self.X.shape[0], dtype=np.float64)
        plotFlag = False#t is not None
        if plotFlag:
            f,ax = plt.subplots(2,sharex=True)
        for kk, cluster in enumerate(self.clusters):
            pi_k = cluster['pi_k']
            mu_k = cluster['mu_k']
            cov_k = cluster['cov_k']
            gamma_nk = (pi_k * gaussian(self.X, mu_k, cov_k)).astype(np.float64)
            if plotFlag:
                ax[0].semilogy(t,gamma_nk,label=Pr(kk))
                # ax[kk].set_ylabel(Pr(kk))
            totals += gamma_nk
            cluster['gamma_nk'] = gamma_nk 
            # print(Pr(kk))
            # print('t:',t[130],t[140])
            # print(gamma_nk[130:140])
        self.totals = totals
        if plotFlag:
            ax[0].semilogy(t,totals,label='total')
            ax[0].legend()
        for kk, cluster in enumerate(self.clusters):
            for i in range(len(totals)):
                if totals[i] == 0.0:
                    cluster['gamma_nk'][i] = 1 / len(self.clusters)
                    totals[i] = 1e-300
                else:
                    cluster['gamma_nk'][i] /= totals[i];
            if plotFlag:
                ax[1].plot(t,cluster['gamma_nk'],label=Pr(kk))
        if plotFlag:
            ax[1].legend()
            ax[0].set_title("Epoch: {0:d}, Likelihood: {1:e}".format(self.epoch, self.get_likelihood()))
            plt.savefig(prefix+"{0:d}.png".format(self.epoch),dpi=600)
            plt.show()
        self.epoch += 1
        if saveFile is not None:
            likelihoods = np.zeros((len(t), len(self.clusters) + 1))
            likelihoods[:,0] = t
            for kk, cluster in enumerate(self.clusters):
                likelihoods[:,kk+1] = cluster['gamma_nk']
            np.savetxt(saveFile, likelihoods)

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
                
            cov_k /= N_k
            # print("Cluster[{0:d}] cond: {1:e}, N_k: {2:e}".format(kk,np.linalg.cond(cov_k), N_k))
            
            cluster['pi_k'] = pi_k
            cluster['mu_k'] = mu_k
            cluster['cov_k'] = cov_k
    def get_likelihood(self):
        sample_likelihoods = np.log(self.totals)
        return np.sum(sample_likelihoods)
    def save(self, meanfile, covarfile, pifile):
        mu0 = np.zeros((len(self.clusters),self.X.shape[1]))
        cov0 = np.zeros((len(self.clusters),self.X.shape[1],self.X.shape[1]))
        pi0 = np.zeros(len(self.clusters))
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
    subset = np.arange(6, 18)
    for key, val in var_idxs.items():
        found_idxs = np.where(subset==val)[0]
        if found_idxs.size > 0:
            var_idxs[key] = found_idxs[0]
        else:
            var_idxs[key] = -1
    n_primitives = 5
    N = len(subset)
    # scaling = 1/np.sqrt(np.genfromtxt("scaling_cov_diag.dat"))
    scaling = np.ones(18)
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
        time, X = read_data1('../data2/run1', '../data2/bias.force',output_fmt='array',tpairlist=tpairs,scale=scaling)
        #each row of X is an observation
        #each column of X is a variable
        mu0[prim.value] = np.mean(X[:,subset],axis=0)
        # print(mu0[prim.value])
        cov0[prim.value] = np.cov(X[:,subset],rowvar=False)
        # cov0[prim.value] = np.diag(np.diag(cov0[prim.value]))
        # print(np.linalg.cond(cov0[prim.value]))
        # print("{0:e}".format(np.linalg.cond(cov0[prim.value])))
    #TRAINING
    time,X = read_data1('../data2/run1', '../data2/bias.force',output_fmt='array',t0=0.0, t1 = 10.5,scale=scaling)
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
    myConstraints[Pr.alignthreads.value] = (
        (var_idxs['vel_x'], 0.0),
        (var_idxs['vel_y'], 0.0),
        (var_idxs['vel_z'], 0.0),
        (var_idxs['ang_vel_x'], 0.0),
        (var_idxs['ang_vel_y'], 0.0),
        (var_idxs['ang_vel_z'], 0.0)
    )
    myGMM = GMM(X[:,subset])
    # myGMM.initialize_clusters(n_primitives, means0=mu0, cov0=cov0)
    myGMM.initialize_clusters(n_primitives, constraints=myConstraints,means0=mu0, cov0=cov0)
    for i in range(20):
        if i == 20:
            myGMM.expectation_step(t=time)
        else:
            myGMM.expectation_step()
        myGMM.maximization_step()
        print("it: {0:d} likelihood  function {1:e}".format(i, myGMM.get_likelihood()))
    myGMM.save('references/mean', 'references/covar', 'references/pi')

    #TESTING
    testfile='../data2/run3'
    print("--------testing: ",testfile, "-----------")
    time,X = read_data1(testfile, '../data2/bias.force',output_fmt='array',t0=0.0, t1 = 10.5,scale=scaling)
    mytestGMM = GMM(X[:,subset])
    mytestGMM.initialize_clusters_from_savedfiles(n_primitives, 'references/mean.npy', 'references/covar.npy', 'references/pi.npy')
    for i in range(20):
        if i == 19:
            mytestGMM.expectation_step(t=time,prefix="figures/run3_",saveFile="run3_likelihoods")
            # mytestGMM.expectation_step(t=time,saveFile="run3_likelihoods")
        else:
            mytestGMM.expectation_step()
        mytestGMM.maximization_step()
        print("it: {0:d} likelihood  function {1:e}".format(i, mytestGMM.get_likelihood()))



