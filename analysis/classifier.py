#!/usr/bin/env python3
import numpy as np
import scipy.linalg
import scipy.stats
import enum
import bisect
import random
import collections
from read_data import read_data0, read_data1, Obs
dt = 0.1

class Pr(enum.Enum): 
    none = 0
    fsm = 1
    contact = 2
    alignthreads = 3
    screw = 4
def sample_primitive(p):
  #p probability distribution of which primitive
  return bisect.bisect(np.cumsum(p), random.random())


# look at the data for each primitive once, run the suggested tests and see which ones yield statistically relevant results, then include those that are statistically relevant in the tree

def initializeTransitionMatrix():
    #transition matrix
    T = np.ones((5,5))
    #
    T[Pr.none.value, Pr.none.value] = 10
    # T[Pr.none.value, Pr.fsm.value] = 0.05
    #
    T[Pr.fsm.value, Pr.fsm.value] = 10
    # T[Pr.fsm.value, Pr.none.value] = 0.05
    # T[Pr.fsm.value, Pr.contact.value] = 0.05
    #
    # T[Pr.contact.value, Pr.fsm.value] = 0.0
    T[Pr.contact.value, Pr.contact.value] = 8
    # T[Pr.contact.value, Pr.alignthreads.value] = 0.2
    #
    T[Pr.alignthreads.value, Pr.alignthreads.value] = 8
    # T[Pr.alignthreads.value, Pr.screw.value] = 0.1
    #
    # T[Pr.screw.value, Pr.none.value] = 0.5
    T[Pr.screw.value, Pr.screw.value] = 10
    T = np.transpose(T.transpose() / np.sum(T,axis=1))
    return T
def forward_model_primitive(s_value, T):
    #s is a primitve idx
    #T is the transition matrix
    return sample_primitive(T[s_value])

def particle_filter(observations,T_forward):
    """ 
    Input:
      observations: states Starting from T=1
      pose_0: (4,4) numpy arrays, starting pose
    Output:
      p_primitives (N x n_primtives probability array)
    """
    N = len(observations)
    n_primitives = 5;
    p_primitives = np.zeros((N, n_primitives))
    N_particles = 100;
    p_primitives[0,Pr.none.value] = 0.9
    p_primitives[0,Pr.fsm.value] = 0.1
    #store primitives as integers
    s = np.zeros((N_particles,N),dtype=int)
    for i in range(N_particles):
      s[i,0] = sample_primitive(p_primitives[0])
    weights = np.ones(N_particles)
    for t in range(N-1):
      print(t)
      ps = observation_model(observations[t])
      for i in range(N_particles):
        s[i,t+1]=forward_model_primitive(s[i,t],T)
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
      for prim_idx in range(n_primitives):
          p_primitives[t+1, prim_idx] = temp[prim_idx]/N_particles
    return p_primitives



if __name__ == "__main__":
    import matplotlib.pyplot as plt
    # from mpl_toolkits.mplot3d import Axes3D
    # from plot_helper import draw_2d, draw_3d
    # from matplotlib import rc
    T=initializeTransitionMatrix()
    t, obs = read_data1('../data2/run1', '../data/bias.force',obs_output=True,t0=0.0,t1=10.0)
    print(len(obs))
    p_primitives = particle_filter(obs,T)
    n_primitives = 5
    for i in range(n_primitives):
      plt.plot(t,p_primitives[:,i],label=Pr(i))
    plt.xlabel('t')
    plt.ylabel('probability')
    plt.legend()
    plt.show()
