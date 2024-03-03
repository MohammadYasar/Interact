import numpy as np
import pandas as pd
import torch

def joint_wise(predicted_data, zeroevel_data):
    predicted_means =  list(); zerovel_means =  list()
    better_joints = list()
    for i in range(predicted_data.shape[1]):
        predicted_means.append(sum(predicted_data[:,i])/len(predicted_data[:,i]))
        zerovel_means.append(sum(zeroevel_data[:,i])/len(zeroevel_data[:,i]))
        if sum(zeroevel_data[:,i])>sum(predicted_data[:,i]):
            better_joints.append(i)

    return predicted_means, zerovel_means, better_joints

def eucdlidean(actual_pose, predicted_pose):
    euc = 0
    print (actual_pose.shape)
    time_steps=  15#predicted_pose.shape[1]
    mse = (actual_pose-predicted_pose)**2
    mse = mse/(10*10)
    mse = mse.reshape(-1, mse.size(-1))
    euc_tensor = torch.FloatTensor(mse.size(0), mse.size(-1)//3)
    for j in range(0,mse.size(-1), 3):

        euc_tensor[:,j//3] = (mse[:,j] + mse[:,j+1] + mse[:,j+2])/3
    euc_tensor = euc_tensor.reshape(-1, time_steps, euc_tensor.size(-1)).permute(1, 0, 2)
    #print (euc_tensor.shape)
    unsqueezed_euc_tensor = (euc_tensor)

    euc_tensor = torch.mean(unsqueezed_euc_tensor, [1,2], True).squeeze()
    #print (euc_tensor)
    print ("& {:.2f} & {:.2f} & {:.2f} & {:.2f} & {:.2f} & {:.2f} & {:.2f}".format(euc_tensor[:2].mean(),euc_tensor[:4].mean(), euc_tensor[:6].mean(),euc_tensor[:8].mean(),euc_tensor[:10].mean(),euc_tensor[:13].mean(),euc_tensor[:15].mean()))
    return euc_tensor,unsqueezed_euc_tensor#.mean()

prediction_file_loc = "/project/CollabRoboGroup/msy9an/GoalPrediction/outputs/csv_files/test_df_preds35.csv"#vistest_preds_epoch0.csv" #ours#ensembletest_preds_epoch106.csv

labels_ours = "/home/msy9an/Skeletonlabels_15_0316.csv"

gt_file_loc = "/project/CollabRoboGroup/msy9an/GoalPrediction/outputs/csv_files/test_gt_epoch0.csv"

prediction_df = np.asarray(pd.read_csv(prediction_file_loc, index_col=0))
gt_df = np.asarray(pd.read_csv(gt_file_loc, index_col=0))
labels_array_ours = np.asarray(pd.read_csv(labels_ours, index_col=0))

eucdlidean(torch.FloatTensor(gt_df), torch.FloatTensor(prediction_df))

"""
labels_dict= dict()
for label in labels_array_ours:
    if label[0] not in labels_dict.keys():
        labels_dict[label[0]] = list()

diff_all = list()
for label in sorted(labels_dict.keys()):
    #print (label)
    index = np.where(labels_array_ours==label)[0]
    diff_all.append(label)
    prediction_category = torch.FloatTensor(prediction_df[index])
    gt_category = torch.FloatTensor(gt_df[index])
    euc_tensor,unsqueezed_euc_tensor = eucdlidean(gt_category, prediction_category)

    diff_all.append(unsqueezed_euc_tensor.mean())
    diff_all.extend(euc_tensor.detach().numpy())

diff_all = np.asarray(diff_all)
diff_all = diff_all.reshape(-1,2+16)
diff_df = pd.DataFrame(diff_all)
diff_df.to_csv("vis_model.csv")
"""
