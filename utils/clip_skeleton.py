import os
import argparse
import cv2
import pandas as pd
import numpy as np
import pickle
import matplotlib.pyplot as plt
from frame_match import *
import datetime
import multiprocessing
import glob, math
# Demo Instruction: python3 plot_skelton.py hhc 2 2 1


def writeSkeleton(corr_indices, natnet_sdk_file, experiment_type,group_num,trial_num):
    
    output_dir = "/scratch/msy9an/data/tro_data/{}/Variation_{}/Group_{}/Trial_{}".format(exp_type, var, group_num, trial_num)
    os.makedirs(output_dir, exist_ok=True)
    output_video_file = "{}/new_processed_skeletons.csv".format(output_dir)
    # if os.path.exists(output_video_file):
    #     return 
    df = pd.read_csv(natnet_sdk_file, index_col=0, header=None, delimiter="\t")
    skeleton_array = np.asarray(df)
    new_skeleton_array = list()
    for i in range(0, len(corr_indices)):
        new_skeleton_array.append(skeleton_array[corr_indices[i][-1]])
    new_skeleton_array = np.asarray(new_skeleton_array, dtype=np.float32)
    print ("before processing: ", new_skeleton_array[0], new_skeleton_array[-1])
    diff = np.mean(np.asarray(new_skeleton_array[10]-new_skeleton_array[-10]))
    if diff == 0.0:
        
        print ("before processing: ",np.mean(np.asarray(skeleton_array[0]-skeleton_array[-1])))
        print ("after processing: ", new_skeleton_array[0] , new_skeleton_array[-1], np.mean(np.asarray(new_skeleton_array[0]-new_skeleton_array[-1])))
        
        print (new_skeleton_array.shape, output_dir)
    new_skeleton_array = pd.DataFrame(new_skeleton_array)
    new_skeleton_array.to_csv(output_video_file)
    return 

def getStartFrames(_file):
    start_frames = np.asarray(pd.read_csv(_file))
    return start_frames
    
def getFileName(experiment_type,variation_num,group_num,trial):
    skel_file =("/project/CollabRoboGroup/datasets/tro_data/{}/Variation_{}/Group_{}/Trial_{}/skeleton_data.csv".format(experiment_type,variation_num,group_num,trial))
    # rgb_file = ("/project/CollabRoboGroup/datasets/tro_data/{}/Variation_{}/Group_{}/Trial_{}/timestamp-withskeleton.csv".format(experiment_type,variation_num,group_num,trial)) #_hhc_{}_{}_{}.csv".format(experiment_type,variation_num,group_num,trial, variation_num,group_num,trial))
    rgb_file = glob.glob("/project/CollabRoboGroup/datasets/tro_data/{}/Variation_{}/Group_{}/Trial_{}/timestamp-withskeleto*".format(experiment_type,variation_num,group_num,trial))[0] #, variation_num,group_num,trial))
    ego_view1 = "/project/CollabRoboGroup/datasets/tro_pupil/HHC-V{}-G{}-I{}-A2/world_timestamps.csv".format(variation_num, group_num, trial)
    ego_view2 = "/project/CollabRoboGroup/datasets/tro_pupil/HHC-V{}-G{}-I{}-A3/world_timestamps.csv".format(variation_num, group_num, trial)
    
    return skel_file, rgb_file, ego_view1, ego_view2


for exp_type in ['HRC']:
    for var in ['1', '2']:
        count = 0    
        _file = "/project/CollabRoboGroup/datasets/tro_data/{}_var{}.csv".format(exp_type.lower(), var)
        start_frames = getStartFrames(_file) #[:]
        # print (int(start_frames[0,2]))
        for i in range(len(start_frames)):
            # print (("start_frames ", start_frames[i,2]))
            if start_frames[i,2] != ' ':
                group_num, trial_num, start_frame = int(start_frames[i,0]), int(start_frames[i,1]), int(start_frames[i,2])
                # try:
                print ("current file ", exp_type,var,group_num,trial_num)
                natnet_ros_file, natnet_sdk_file, ego_view1, ego_view2 = getFileName(exp_type,var,group_num,trial_num)
                natnet_sdk_array = read_sdkcsv(natnet_sdk_file, delim='\t')
                natnet_ros_array = read_roscsv(natnet_ros_file)[start_frame:]
                corr_indices = loop_queryarray(natnet_ros_array, natnet_sdk_array)
                writeSkeleton(corr_indices, natnet_sdk_file, exp_type, group_num, trial_num)
                                
                # except:
                #     print ("no file found ", exp_type,var,group_num,trial_num)