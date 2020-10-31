"""======================================================================================
 plotresults.py
 
 Input: runs you want to run and num of Tmatrix updates    
 Output: Plots segmented data both manual and automatic

 *note -> likelihood plots are created and saved inside 
       the expectation_step called by train and test in gmm.py
 
Last update, Fall 2020
======================================================================================"""
numTMatrixUpdates = 15 
lastT = numTMatrixUpdates - 1
print(">>>> Pato dibujando")

run2plot = [2, 6, 18]
trans2plot = [0,lastT]

"""
CAP

"""
# ----------------------------------
# Plot sensor data of labelled run 
#   - for initial and final transition matrix
#   - for run2 (really good) and run12 (sucks)
#   - 4 plots     
# run2plot = [2, 12]
# for i in range(2): 
#     for t in range(2):
#         plot_file('../data/medium_cap/raw_medium_cap/run{0:d}'.format(run2plot[i]),
#             tlabelfile="results/run{0:d}_tlabels_T{1:d}".format(run2plot[i],trans2plot[t]),
#             prlabelfile="results/run{0:d}_prmlabels_T{1:d}".format(run2plot[i],trans2plot[t]),
#             tlabelfileTruth='../data/medium_cap/raw_medium_cap/run{0:d}_tlabels'.format(run2plot[i]),
#             prlabelfileTruth='../data/medium_cap/raw_medium_cap/run{0:d}_prmlabels'.format(run2plot[i])
#             )
#         plt.savefig("figures/labelled_run{0:d}_T{1:d}.png".format(run2plot[i],trans2plot[t]),dpi=600)
#         # plt.show()
#         plt.close()


"""
PIPE

"""     
run2plot = [2, 12]
trans2plot = 0

for i in range(2): 
    plot_file('../data/pipe/raw_pipe/run{0:d}'.format(run2plot[i]),
        tlabelfile="results/run{0:d}_tlabels_T{1:d}".format(run2plot[i],trans2plot),
        prlabelfile="results/run{0:d}_prmlabels_T{1:d}".format(run2plot[i],trans2plot)
#       tlabelfileTruth='../data/pipe/raw_pipe/run{0:d}_tlabels'.format(run2plot[i]),
#       prlabelfileTruth='../data/pipe/raw_pipe/run{0:d}_prmlabels'.format(run2plot[i])
        )
    plt.savefig("figures/labelled_run{0:d}_T{1:d}.png".format(run2plot[i],trans2plot),dpi=600)
    plt.close()

print(">>>> Pato ha terminado de dibujar")