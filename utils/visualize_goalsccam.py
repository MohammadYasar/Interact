import os, sys, glob, scipy, h5py, scipy.io
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.animation as animation

csv_path = "/project/CollabRoboGroup/msy9an/data/CCAM/realsense/yq1.csv"


csv_dict = dict()

gt_csv_path = "/project/CollabRoboGroup/msy9an/GoalPrediction/outputs/csv_files/test_gt_epoch0.csv"
preds_csv_path = "/project/CollabRoboGroup/msy9an/GoalPrediction/outputs/csv_files/test_df_preds10.csv"

subject = "capps1"
# gt_csv_path = "/project/CollabRoboGroup/msy9an/data/CCAM/{}.csv".format(subject)





def check_data(csv_path, subject):
    pred_labels_path = "/project/CollabRoboGroup/msy9an/GoalPrediction/src/predictions.csv"
    predicted_labels = pd.read_csv(pred_labels_path, index_col=0)
    predicted_labels = np.asarray(predicted_labels).reshape(-1,2)
    print (csv_path)
    df = pd.read_csv(csv_path, index_col=0)
    data = np.asarray(df)[:,9+12:12+12]
    
    labels = np.asarray(df)[:,-1].reshape(-1,1)
    print (predicted_labels, data.shape)
    colors = ['r', 'g', 'b', 'yellow', 'orange']
    fig = plt.figure()
    ax1 = fig.add_subplot(111, projection='3d')
    print ("df shape ", data.shape, df.shape)
    for i in range(data.shape[0]-30):
        ax1.plot(data[i:i+1,0], data[i:i+1,2], data[i:i+1,1], colors[int(labels[i][0])-1], marker=".", markersize=4, lw=2)

    plt.axis('on')
    plt.grid(b=None)
    #plt.pause(.1)
    plt.savefig("/project/CollabRoboGroup/msy9an/data/mo_pred_visualization/gt{}.png".format(subject))
    
    for i in range(data.shape[0]-30):
        ax1.plot(data[i:i+1,0], data[i:i+1,2], data[i:i+1,1], colors[int(predicted_labels[i][0])], marker="x", markersize=4, lw=2)


    plt.axis('on')
    plt.grid(b=None)
    #plt.pause(.1)
    plt.savefig("/project/CollabRoboGroup/msy9an/data/mo_pred_visualization/preds{}.png".format(subject))
    return 

check_data(csv_path, subject)

#ls_x,ls_y,ls_z,rs_x,rs_y,rs_z,le_x,le_y,le_z,re_x,re_y,re_z,lw_x,lw_y,lw_z,rw_x,rw_y,rw_z,lh_x,lh_y,lh_z,rh_x,rh_y,rh_z
"""
df = pd.read_csv(gt_csv_path, index_col=0)
ground_truth = np.asarray(df).reshape(df.shape[0], 8, 3)
ground_truth = np.transpose(ground_truth, (2,1,0))
ground_truth = np.transpose(ground_truth, (1,0,2))

df = pd.read_csv(preds_csv_path, index_col=0)
predictions = np.asarray(df).reshape(df.shape[0], 8, 3)
predictions = np.transpose(predictions, (2,1,0))
predictions = np.transpose(predictions, (1,0,2))

fig = plt.figure()
fig.tight_layout()
ax1 = fig.add_subplot(111, projection='3d')
ax1.view_init(75)
ax1.set_xlim(-3,3)
ax1.set_ylim(0.5,1.5)
ax1.set_zlim(-5,5)


mse = 100*100*(ground_truth-predictions)**2
print ("mse: ", mse.mean())

#ground_truth = (ground_truth.transpose())

#predictions = (predictions.transpose())
print (ground_truth.shape)

frame = 100#2000
fig = plt.figure()

joints = [1, 3, 2, 4, 3, 5, 4, 6, 5, 7, 6, 8]

joints = [j-1 for j in joints]


cluster_counter = 0
for i in range(0,100, 1):
    #ax2 = fig.add_subplot(212, projection='3d')
    #ax3 = fig.add_subplot(313, projection='3d')
    ax1 = fig.add_subplot(111, projection='3d')
    ax1.view_init(0)
    #print (i)
    
    ax1.view_init(0)
    for index in range(0,len(joints),2):
    
        
        gt_x = (ground_truth[joints[index:index+2], 0, i])
        gt_y = (ground_truth[joints[index:index+2], 1, i])
        gt_z = (ground_truth[joints[index:index+2], 2, i])
        #print (x)

        ax1.plot(gt_x, gt_y, gt_z, 'r', marker=".", markersize=4, lw=2)#, alpha=0.0)
        """"""
        x = (predictions[joints[index:index+2], 0, i])
        y = (predictions[joints[index:index+2], 1, i])
        z = (predictions[joints[index:index+2], 2, i])
        ax1.plot(x, y, z, 'grey', marker=".", markersize=4, lw=2)



    #print (pck/count)
    plt.axis('on')
    plt.grid(b=None)
    #plt.pause(.1)
    plt.savefig("/project/CollabRoboGroup/msy9an/data/CCAM/mo_pred_visualization/prediction_plots{}.png".format(i))

"""