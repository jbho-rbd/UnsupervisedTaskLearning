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
    align = 3
    screw = 4
def sample_primitive(p):
  #p probability distribution of which primitive
  return bisect.bisect(np.cumsum(p), random.random())

#assume all tests are binary
#then we have a number which represents in binary which ones are passed
#and an array which represents all the probabilities (2^n)
#for each primitive -- you see what tests you need to make

# look at the data for each primitive once, run the suggested tests and see which ones yield statistically relevant results, then include those that are statistically relevant in the tree

def initializeTransitionMatrix():
    #transition matrix
    T = np.zeros((5,5))
    #
    T[Pr.none.value, Pr.none.value] = 0.95
    T[Pr.none.value, Pr.fsm.value] = 0.05
    #
    T[Pr.fsm.value, Pr.fsm.value] = 0.9
    T[Pr.fsm.value, Pr.none.value] = 0.05
    T[Pr.fsm.value, Pr.contact.value] = 0.05
    #
    T[Pr.contact.value, Pr.fsm.value] = 0.0
    T[Pr.contact.value, Pr.contact.value] = 0.8 
    T[Pr.contact.value, Pr.align.value] = 0.2
    #
    T[Pr.align.value, Pr.align.value] = 0.8
    T[Pr.align.value, Pr.screw.value] = 0.1
    #
    T[Pr.screw.value, Pr.none.value] = 0.5
    T[Pr.screw.value, Pr.screw.value] = 0.5
    T = np.transpose(T.transpose() / np.sum(T,axis=1))
    return T
def forward_model_primitive(s_value, T):
    #s is a primitve idx
    #T is the transition matrix
    return sample_primitive(T[s_value])
def likelihood(x, mean, stdev):
    return 2*scipy.stats.norm.cdf(-abs((x- mean)/stdev))
def observation_tests(x):
    dims = range(3)
    ps = np.zeros(6)
    #test 0 - are the velocities zero
    p = 1
    for i in dims:
      p = p*likelihood(x.vel[i],0.0, 0.2)
    ps[0] = p
    #test 1 - are the forces zero
    p = 1
    for i in dims:
      p = p*likelihood(x.F[i],0.0, 2.0)
    ps[1] = p
    #test 2 - are the moments zero
    p = 1
    for i in dims:
      p = p*likelihood(x.M[i],0.0, 0.1)
    ps[2] = p
    #test 3 - are the angular velocities zero
    p = 1
    for i in dims:
      p = p*likelihood(x.ang_vel[i],0.0, 1)
    ps[3] = p
    #test 4 - is omega_z zero
    p = likelihood(x.ang_vel[2],0.0, 0.3)
    ps[4] = p
    #test 5 - is M_z zero
    p = likelihood(x.M[2],0.0, 0.01)
    ps[5] = p
    return ps
def observation_model(x):
    #s the underlying primitive state
    #x the observation: velocities positions etc...
    #returns the probability of observing x, given s, for n different s
    p = observation_tests(x)
    ps = np.array([p[0]*p[2],
                  (1 - p[0])*p[2],
                  (1 - p[0])*(1 - p[2]),
                  p[0]*(1 - p[2])*p[4],
                  p[0]*(1 - p[2])*(1 - p[4])])
    return ps

def exp_coords_6D_from_4x4_pose(pose):
    temp = np.real(scipy.linalg.logm(pose))
    return np.array([temp[0, 3], temp[1, 3], temp[2, 3], temp[2,1], temp[0, 2], temp[1,0]])

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
