#!/usr/bin/env python3
import numpy as np
import scipy.linalg
import scipy.stats
import enum
import bisect
import random
import collections
dt = 0.1

class Pr(enum.Enum): 
    none = 0
    fsm = 1
    contact = 2
    align = 3
    screw = 4
class Obs():
    def __init__(self, pos, ori, vel, ang_vel, force, moment):
        self.pos = pos
        self.ori = ori
        self.vel = vel
        self.ang_vel = ang_vel
        self.F = force
        self.M = moment
def sample_primitive(p):
  #p probability distribution of which primitive
  return Pr(bisect.bisect(np.cumsum(p), random.random()))

def initializeTransitionMatrix():
    #transition matrix
    T = np.zeros((5,5))
    #
    T[Pr.none.value, Pr.none.value] = 0.2
    T[Pr.none.value, Pr.fsm.value] = 0.5
    #
    T[Pr.fsm.value, Pr.fsm.value] = 0.8
    T[Pr.fsm.value, Pr.contact.value] = 0.2
    #
    T[Pr.contact.value, Pr.fsm.value] = 0.2
    T[Pr.contact.value, Pr.contact.value] = 0.8 
    T[Pr.contact.value, Pr.align.value] = 0.1
    #
    T[Pr.align.value, Pr.contact.value] = 0.8
    T[Pr.contact.value, Pr.contact.value] = 0.1
    #
    T[Pr.screw.value, Pr.none.value] = 0.5
    T[Pr.screw.value, Pr.screw.value] = 0.5
    T = np.transpose(T.transpose() / np.sum(T,axis=1))
    s = Pr.fsm
    print(Pr(Pr.contact.value))
    return (T)
def forward_model_primitive(s_value, T):
    #s is a primitve idx
    #T is the transition matrix
    return sample_primitive(T[s_value])
def likelihood(x, mean, stdev):
    return 1 - 2*scipy.stats.normal.cdf(-fabs((x- mean)/stdev))
def observation_model(s,x):
    #s the underlying primitive state
    #x the observation: velocities positions etc...
    #returns the probability of observing x, given s
    if s == Pr.none:
      #forces, moments, and velocities should be close to zero
      p = 1
      forceNoise = 1;#stdev
      momentNoise = 0.1;#stdev
      velNoise = 1;#stdev
      dims = range(3)
      for i in dims:
        p = p*likelihood(x.F[i],0.0,forceNoise)
      for i in dims:
        p = p*likelihood(x.M[i],0.0,momentNoise)
      for i in dims:
        p = p*likelihood(x.vel[i],0.0,velNoise)
      return 1 - p
    elif s == Pr.fsm:
      #linear velocities are proportional to displacement
      #forces and moment are small
      p = 1
      forceNoise = 1;#stdev
      momentNoise = 0.1;#stdev
      velNoise = 10;#stdev
      k = 3 #some kind of gain
      dims = range(3)
      for i in dims:
        p = p*likelihood(x.F[i],0.0,forceNoise)
      for i in dims:
        p = p*likelihood(x.M[i],0.0,momentNoise)
      for i in dims:
        p = p*likelihood(x.vel[i], k*x.pos[i], velNoise)
      return p
    elif s == Pr.contact:
      #zero velocity
      #high forces (relative to displacement)
      print("contact not implemented")
      exit()
    elif s == Pr.align:
      #very low, zero velocities of all kinds
      #high Z force
      p = 1
      forceNoise = 1;#stdev
      momentNoise = 0.1;#stdev
      velNoise = 1;#stdev
      angvelNoise = 100;#xstdev
      dims = range(3)
      for i in [0, 1]:
        p = p*likelihood(x.F[i],0.0,forceNoise)
      p = p*likelihood(x.F[2], 10.0, i)
      for i in dims:
        p = p*likelihood(x.M[i],0.0,momentNoise)
      for i in dims:
        p = p*likelihood(x.vel[i],0.0,velNoise)
      for i in dims:
        p = p*likelihood(x.vel[i],0.0,angvelNoise)
      return 1 - p
    elif s == Pr.screw:
      print("screw not implemented")
      exit()
    else:
      print("primitive not recognized")
      exit()
def exp_coords_6D_from_4x4_pose(pose):
    temp = np.real(scipy.linalg.logm(pose))
    return np.array([temp[0, 3], temp[1, 3], temp[2, 3], temp[2,1], temp[0, 2], temp[1,0]])

def particle_filter(observations):
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
    p0 = np.zeros(5);
    p0[Pr.none.value] = 0.99
    p0[Pr.fsm.value] = 0.01
    #store primitives as integers
    s = np.zeros((N_particles,N),dtype=int)
    for s in range(N_particles):
      s[i,0] = sample_primitive(p0).value
    weights = np.ones(N_particles)
    T = initializeTransitionMatrix()
    for t in range(N):
      print(t)
      for i in range(N_particles):
        s[i,t+1] = forward_model_primitive(s[i,t],0)
        poses[i,t+1] = forward_model_pose(twists[i,t], poses[i,t])
        weights[i] = observation_model(s,observations[t])
      #normalize weights
      print(np.mean(weights))
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
      print(sum(weights), n_particles_allocated)
      #count primtivies
      temp = collections.Counter(s[:,t])
      for prim_idx in range(n_particles):
          p_primitives[t, prim_idx] = temp[prim_idx]/N_particles
    return p_primitives



if __name__ == "__main__":
    # import matplotlib.pyplot as plt
    # from mpl_toolkits.mplot3d import Axes3D
    # from plot_helper import draw_2d, draw_3d
    # from matplotlib import rc
    initializeTransitionMatrix()
