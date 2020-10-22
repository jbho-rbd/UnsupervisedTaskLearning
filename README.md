# rubberDucky1
Hi I'm a pato

# Contents 
1) Analysis: 
    ** Key files
        gmm.py - used to identify and segment human demonstration data into primitives. 
               - uses read_data and plot_data
        read_data.py - reads in raw sensor data and processes it (change of frame, offsets, quaternions...)
        testmultiple.sh - bash script to run gmm on many runs at once
    ** Ploting 
        plot_data.py - plots the sensor data with labels 
        plotconfusion.py - plots a confusion matrix to visualize labelling errors
        plotbar.py - plots a bar diagram that respresents the same concept as the confusion matrix
        plotbox.py - makes a box plot with the mean and standard deviation of all runs
    ** Old or test files
       analyse.py
       classifier.py
       jupyter notebooks - used for plot debugging
       test.py - uses classifier.py 
    
2) Data: 
    - human demonstrations of complete manipulation tasks
    - manipulated object pose and contact forces and moments (recorded using optitrack and a 6DOF optoforce sensor)
    
3) Training: 
    network.py - learns a transition model for the task based on the data labelled by the gmm
