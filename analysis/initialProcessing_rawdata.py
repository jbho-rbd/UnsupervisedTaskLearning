"""======================================================================================
processingAndlabelling_rawdata.py
 
Script that uses read_data and plot_data to process 
the raw sensor measurments after being collected

- run as: 
python processing_manual_labelling_rawdata.py run_number_start experiment_name

- where: 
run_number_start = first run you want to make a plot of
experiment_name = cap, pipe or bulb

======================================================================================"""

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from scipy.spatial.transform import Rotation as R
from matplotlib import rc, rcParams
from read_data import read_data1
from plot_data import plot_file
from classifier import Pr
import sys
from shutil import copyfile

# latex formatting
rc('text',usetex=True)
rcParams['axes.titlesize'] = 'x-large'
rcParams['axes.labelsize'] = 'large'
rcParams['xtick.labelsize'] = 'x-large'
rcParams['ytick.labelsize'] = 'x-large'

# inputs and constants
run_number_start = int(sys.argv[1])
experiment_name = sys.argv[2]
NUM_RUNS = 21 

def on_plot_hover(event):
    # Iterating over each data member plotted
    for curve in plt.get_lines():
        # Searching which data member corresponds to current mouse position
        if curve.contains(event)[0]:
            print "over %s" % curve.get_gid()

# 1- process raw data (read_data) and make plots to see how to split it manually
# TODO just edit the names of the directories based on the cmd input instead of copying the code
print('patito dibujando')
if experiment_name == 'pipe':
	print('pipe')
	NUM_RUNS = 26 
	for run_number in range(run_number_start, NUM_RUNS): 
	    print('run {0:d}'.format(run_number))       
	    plot_file('../data/pipe/raw_pipe/run{0:d}_pipe'.format(run_number))
	    plt.savefig("figures2label/pipe/run{0:d}_pipe_raw.png".format(run_number),dpi=600, bbox_inches = 'tight',pad_inches = 0)
	    plt.close()

if experiment_name == 'medium_cap':
	print('medium_cap')
	NUM_RUNS = 26
	for run_number in range(run_number_start, NUM_RUNS): 
		if run_number == 11:
			continue
		print('run {0:d}'.format(run_number))       
		plot_file('../data/medium_cap/raw_medium_cap/run{0:d}'.format(run_number))
		plt.savefig("figures2label/medium_cap/run{0:d}_cap_raw.png".format(run_number),dpi=600, bbox_inches = 'tight',pad_inches = 0)
		plt.close()

if experiment_name == 'big_cap':
	print('big_cap')
	for run_number in range(run_number_start, NUM_RUNS): 
	    print('run {0:d}'.format(run_number))       
	    plot_file('../data/big_cap/raw_big_cap/run{0:d}'.format(run_number))
	    plt.savefig("figures2label/big_cap/run{0:d}_cap_raw.png".format(run_number),dpi=600, bbox_inches = 'tight',pad_inches = 0)
	    plt.close()

if experiment_name == 'small_cap':
	print('small_cap')
	NUM_RUNS = 22
	for run_number in range(run_number_start, NUM_RUNS): 
	    print('run {0:d}'.format(run_number))       
	    plot_file('../data/small_cap/raw_small_cap/run{0:d}'.format(run_number))
	    plt.savefig("figures2label/small_cap/run{0:d}_cap_raw.png".format(run_number),dpi=600, bbox_inches = 'tight',pad_inches = 0)
	    plt.close()

if experiment_name == 'bulb':
	print('bulb')
	for run_number in range(run_number_start, NUM_RUNS): 
	    print('run {0:d}'.format(run_number))       
	    plot_file('../data/bulb/raw_bulb/run{0:d}_bulb'.format(run_number))
	    plt.savefig("figures2label/bulb/run{0:d}_bulb_raw.png".format(run_number),dpi=600, bbox_inches = 'tight',pad_inches = 0)
	    plt.close()

if experiment_name == 'bulb1':
	print('bulb1')
	for run_number in range(run_number_start, NUM_RUNS): 
	    print('run {0:d}'.format(run_number))       
	    plot_file('../data/bulb/raw_bulb_1/run{0:d}'.format(run_number))
	    plt.savefig("figures2label/bulb_1/run{0:d}_bulb_raw1.png".format(run_number),dpi=600, bbox_inches = 'tight',pad_inches = 0)
	    plt.close()

# print("patito eligiendo tiempos")

# if experiment_name == 'cap':
# 	run_number = 1
# 	# 2- Choose split times and labels and generate run#_tlabels and run#_prmlabels
# 	vlines=np.array([0.0, 0.84, 3.27, 4.21, 5.07, 6.74, 6.93, 7.44, 7.588, 8.06, 8.20, 8.68, 8.89, 9.56, 10.36])
# 	np.savetxt("../data/medium_cap/raw_medium_cap/run{0:d}_tlabels_test".format(run_number),vlines[1:])
# 	labels = [Pr.none, Pr.fsm, Pr.align, Pr.engage, Pr.screw,  Pr.none, Pr.screw, Pr.none, Pr.screw, Pr.none, Pr.screw, Pr.none, Pr.screw, Pr.tighten]
# 	np.savetxt("../data/medium_cap/raw_medium_cap/run{0:d}_prmlabels_test".format(run_number),[label.value for label in labels])
# 	# 3- make plot to see if the manually selected labels look reasonable
# 	plot_file('../data/medium_cap/raw_medium_cap/run{0:d}'.format(run_number),
# 		tlabelfile="../data/medium_cap/raw_medium_cap/run{0:d}_tlabels_test".format(run_number),
# 		prlabelfile="../data/medium_cap/raw_medium_cap/run{0:d}_prmlabels_test".format(run_number))
# 	plt.savefig("figures2label/cap/run{0:d}_raw_labelTest.png".format(run_number),dpi=600, bbox_inches = 'tight',pad_inches = 0)
# 	plt.close()

# if experiment_name == 'pipe':
# 	run_number = 1
	# "interactive" angular velocities plot with tooltip to give x,y values when hovering over a line on the plot
	# t, p1, vels, euler, omegas, F, M = read_data1('../data/pipe/raw_pipe/run{0:d}_pipe'.format(run_number))
	# fig = plt.figure()
	# # plt = fig.add_subplot(111)
	# plt.plot(t,omegas[:,0],'r',label='$\omega_x$')
	# plt.plot(t,omegas[:,1],'g',label='$\omega_y$')
	# plt.plot(t,omegas[:,2],'b',label='$\omega_z$')
	# # plt.set_ylabel('rad/s')
	# fig.canvas.mpl_connect('motion_notify_event', on_plot_hover)           
	# plt.show()
	# 2- Choose split times and labels and generate run#_tlabels and run#_prmlabels
	# vlines=np.array([0.0, 1.2, 4.00, 5.65, 6.38,
	# 	7.11, 7.38, 9.03, 10.43, 11.19, 11.83, 12.36, 13.09, 13.76, 14.60, 15.52, 15.97,
	# 	17,18.5])
	# vlines=np.array([0.0, 1.2, 4.00, 5.65, 6.38,
	# 	7.11, 7.38, 9.03, 9.75, 10.43, 11.19, 11.83, 12.36, 13.09, 13.76, 14.60, 15.52, 
	# 	17,18.5])
	# np.savetxt("../data/pipe/raw_pipe/run{0:d}_tlabels".format(run_number),vlines[1:])
	# labels = [Pr.none, Pr.fsm, Pr.align, Pr.engage, 
	# 	Pr.screw, Pr.none, Pr.screw, Pr.none, Pr.screw, Pr.none, Pr.screw, Pr.none, Pr.screw, Pr.none, Pr.screw, Pr.none, 
	# 	Pr.tighten, Pr.none] #17
	# np.savetxt("../data/pipe/raw_pipe/run{0:d}_prmlabels".format(run_number),[label.value for label in labels])
	# # 3- make plot to see if the manually selected labels look reasonable
	# plot_file('../data/pipe/raw_pipe/run{0:d}_pipe'.format(run_number),
	# 	tlabelfile="../data/pipe/raw_pipe/run{0:d}_tlabels".format(run_number),
	# 	prlabelfile="../data/pipe/raw_pipe/run{0:d}_prmlabels".format(run_number))
	# plt.savefig("figures2label/pipe/run{0:d}_raw_labels.png".format(run_number),dpi=600, bbox_inches = 'tight',pad_inches = 0)
	# plt.close()

print("patito ha terminado!")



# process raw data

# interactive plot

# select labels and plot