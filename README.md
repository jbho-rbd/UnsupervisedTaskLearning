# rubberDucky1
Hi I'm a pato. Make sure you have python3.

# Contents 
## Analysis: 
   - Directories: 
      - /figures
      - /figures2label
      - /references
      - /results
      - /transitions
      - /past_results
   - Key files
      - gmm.py and gmm_objects.py
         - used to identify and segment human demonstration data into primitives
         - gmm_objects simplifies the use on data from different objects
         - method: Hidden Markov Model with particle filter
         - uses: read_data.py and plot_data.py
         - requires: having created 3 directories called: transitions, results, figures, references.
      - initialProcessing_rawdata.py 
         - script for use right after collecting sensor measurements
         - uses: read_data.py and plot_data.py
         - see file for how to use and functions
      - finalProcessing_results.py
         -  plots segmented data both manual and automatic by using functions in plot_data 
         -  computes success rate as the similarity between demonstrated and labelled primitive sequences
      - read_data.py - reads in raw sensor data and processes it (change of frame, offsets, quaternions...)
      - plot_data.py (pending refactoring)
          - plot_file() - plots the segmented sensor data with labels
          - getlabels()
          - compute_success_rate() - really it's computing accuracy as defined in the paper
          - write_Pr_file()
      - testmultiple.sh - bash script to run gmm.py on many runs at once
   - Ploting 
        - plotconfusion.py - plots a confusion matrix to visualize labelling errors
        - plotbar.py - plots a bar diagram that respresents the same concept as the confusion matrix
        - plotbox.py - makes a box plot with the mean and standard deviation of all runs
        - plotTmatrixdata.py - 3x1 plot containing data about Tmatrix updates
   - Old or test files
       - analyse.py
       - classifier.py
       - jupyter notebooks - used for plot debugging
       - test.py - uses classifier.py 
    
## Data: 
  - human demonstrations of complete manipulation tasks
  - manipulated object pose and contact forces and moments (recorded using optitrack and a 6DOF optoforce sensor)
  - objects: small, medium, large bottle, lightbulb, pipe
    
## Training: 
  - network.py 
  - learns a transition model for the task based on the data labelled by the gmm
