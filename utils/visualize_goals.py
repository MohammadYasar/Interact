import os, sys, glob, scipy, h5py, scipy.io
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

csv_path = "/project/CollabRoboGroup/msy9an/data/GoalPrediction/Dry_Run/*.csv"


csv_dict = dict()

for csv_file in sorted(glob.glob(csv_path)):
    csv_df = np.array(pd.read_table(csv_file))
    print(csv_df.shape)
    subject = csv_file.split("/")[-1].replace(".csv","")
    
    new_csv = list()
    
    for i in range(csv_df.shape[0]):
        new_csv.append(str(csv_df[i]).split(",")[26:29])
    
    new_csv = np.asarray(new_csv).reshape(len(new_csv),-1)[7:]
    csv_df = np.zeros((new_csv.shape[0], new_csv.shape[1]))
    for i in range(new_csv.shape[0]):
        for j in range(new_csv.shape[1]):
            if new_csv[i][j] != "":
                csv_df[i][j] = new_csv[i][j].astype(np.float)
            else:
                "print (new_csv[i][j])"
            
    csv_dict[subject] = np.asarray(csv_df)
    print (csv_df.shape)
    csv_dict[subject] = csv_dict[subject][csv_dict[subject]!=0.0].reshape(-1,3)
    print (csv_dict[subject].shape)

    fig = plt.figure()
    fig.tight_layout()
    ax1 = fig.add_subplot(111, projection='3d')
    ax1.view_init(30)
    ax1.set_xlim(-3,3)
    ax1.set_ylim(0.5,1.5)
    ax1.set_zlim(-5,5)
    
    # for key in csv_dict.keys():
        
    
    demonstration = csv_df # csv_dict[key]
    
    ax1.plot((demonstration[:,0]), (demonstration[:,1]), (demonstration[:,2]))    

    plt.legend()
    plt.savefig("/project/CollabRoboGroup/msy9an/data/GoalPrediction/combinedgoal_plots_{}.png".format(subject))
    plt.close()
    
"""       
 
 
gt_csv_path = "/project/CollabRoboGroup/msy9an/GoalPrediction/outputs/csv_files/test_gt_epoch0.csv"
preds_csv_path = "/project/CollabRoboGroup/msy9an/GoalPrediction/outputs/csv_files/test_df_preds0.csv"

ground_truth = np.array(pd.read_csv(gt_csv_path, index_col=0)).reshape(-1, 3)
predictions = np.array(pd.read_csv(preds_csv_path, index_col=0)).reshape(-1, 3)

fig = plt.figure()
fig.tight_layout()
ax1 = fig.add_subplot(111, projection='3d')
ax1.view_init(75)
ax1.set_xlim(-3,3)
ax1.set_ylim(0.5,1.5)
ax1.set_zlim(-5,5)


mse = 100*100*(ground_truth-predictions)**2
print ("mse: ", mse.mean())
for i in range(0, predictions.shape[0],15):
    ax1.plot(ground_truth[i:i+1,0], ground_truth[i:i+1,1], ground_truth[i:i+1,2], 'red', marker=".", markersize=1, lw=1)
    
    ax1.plot(predictions[i:i+1,0], predictions[i:i+1,1], predictions[i:i+1,2], 'blue', marker=".", markersize=1, lw=1)
plt.axis('on')
plt.grid(b=None)
#plt.pause(.1)
plt.savefig("/project/CollabRoboGroup/msy9an/data/GoalPrediction/prediction_plots2.png")
    

"""