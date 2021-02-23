"""======================================================================================
initialProcessing_rawdata.py
 
Script that uses read_data and plot_data to process 
the raw sensor measurments after being collected

HOWTO RUN FILE: 
	- python initialProcessing_rawdata.py run_number_start experiment_name
	- comment out whatever function you don't want to use in MAIN

INPUTS: 
	- run_number_start = first run you want to make a plot of
	- experiment_name = medium_cap, pipe or bulb

OUTPUTS:
	- plots of sensor data without vertical segmentation lines
	- saves vlines and labels to corresponding /data folder
	- plots sensor data with vertical lines according to manually selected vlines and labels


Elena, Feb 2021
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

#constants
NUM_RUNS = 21 

# cmd inputs 
run_number_start = int(sys.argv[1])
experiment_name = sys.argv[2]


""" --------------------------------------------------------------------------------------
   Utility Functions
----------------------------------------------------------------------------------------"""
def on_plot_hover(event):
    # Iterating over each data member plotted
    for curve in plt.get_lines():
        # Searching which data member corresponds to current mouse position
        if curve.contains(event)[0]:
            print "over %s" % curve.get_gid()

def process_raw_data(experiment_name):
	print('patito dibujando raw data')
	print(experiment_name)	
	for run_number in range(run_number_start, NUM_RUNS): 
		if experiment_name == 'medium_cap':
			if run_number == 11:
					continue	
		print('run {0:d}'.format(run_number))       
		plot_file('../data/{0}/raw_{1}/run{2:d}'.format(experiment_name, experiment_name, run_number))
		plt.savefig("figures2label/{0}/run{1:d}_{2}_raw.png".format(experiment_name,run_number, experiment_name),
	    	dpi=600, bbox_inches = 'tight',pad_inches = 0)
		plt.close()

def interactive_omega_plot(experiment_name, run_number): 
	'''
	"interactive" angular velocities plot with tooltip 
	to give x,y values when hovering over a line on the plot
	'''
	print("patito eligiendo tiempos")
	t, p1, vels, euler, omegas, F, M = read_data1('../data/{0}/raw_{1}/run{2:d}'.format(experiment_name, experiment_name, run_number))
	fig = plt.figure()
	# plt = fig.add_subplot(111)
	plt.plot(t,omegas[:,0],'r',label='$\omega_x$')
	plt.plot(t,omegas[:,1],'g',label='$\omega_y$')
	plt.plot(t,omegas[:,2],'b',label='$\omega_z$')
	# plt.set_ylabel('rad/s')
	fig.canvas.mpl_connect('motion_notify_event', on_plot_hover)           
	plt.show()

def manual_labels_plot(experiment_name, run_number):
	#make plot to see if the manually selected labels look reasonable
	plot_file('../data/{0}/raw_{1}/run{2:d}'.format(experiment_name, experiment_name, run_number),
		tlabelfile="../data/{0}/raw_{1}/run{2:d}_tlabels_test".format(experiment_name, experiment_name, run_number),
		prlabelfile="../data/{0}/raw_{1}/run{2:d}_prmlabels_test".format(experiment_name, experiment_name, run_number))
	plt.savefig("figures2label/{0}/run{1:d}_raw_labelTest.png".format(experiment_name, run_number),
		dpi=600, bbox_inches = 'tight',pad_inches = 0)
	plt.close()

def save_manual_labels(experiment_name, run_number, vlines, labels):
	np.savetxt("../data/{0}/raw_{1}/run{2:d}_tlabels".format(experiment_name, experiment_name,run_number),
		vlines[1:])
	np.savetxt("../data/{0}/raw_{1}/run{2:d}_prmlabels".format(experiment_name, experiment_name,run_number),
		[label.value for label in labels])

""" --------------------------------------------------------------------------------------
   MAIN
-----------------------------------------------------------------------------------------"""
if __name__ == "__main__":

	# Manual labels

	# --- CAP
	# run 1
	vlines=np.array([0.0, 0.84, 3.27, 4.21, 5.07, 6.74, 6.93, 7.44, 7.588, 8.06, 8.20, 8.68, 8.89, 9.56, 10.36])
	labels = [Pr.none, Pr.fsm, Pr.align, Pr.engage, Pr.screw,  Pr.none, Pr.screw, Pr.none, Pr.screw, Pr.none, Pr.screw, Pr.none, Pr.screw, Pr.tighten]
	
	# --- PIPE
	# run 1
	vlines=np.array([0.0, 1.2, 4.00, 5.65, 6.38,
		7.11, 7.38, 9.03, 9.75, 10.43, 11.19, 11.83, 12.36, 13.09, 13.76, 14.60, 15.52, 
		17,18.5])
	labels = [Pr.none, Pr.fsm, Pr.align, Pr.engage, 
		Pr.screw, Pr.none, Pr.screw, Pr.none, Pr.screw, Pr.none, Pr.screw, Pr.none, Pr.screw, Pr.none, Pr.screw, Pr.none, 
		Pr.tighten, Pr.none] #17

	# 1- process raw data (read_data) and make plots to see how to split it manually
	process_raw_data(experiment_name)

	# 2- interactive plot of one sensor variable
	# interactive_omega_plot(experiment_name, run_number_start)

	# 3- select labels and and save 
	# save_manual_labels(experiment_name, run_number_start, vlines, labels)

	# 4 - plot manual segmentation
	# manual_labels_plot(experiment_name, run_number_start)

	print("patito ha terminado initial processing!")