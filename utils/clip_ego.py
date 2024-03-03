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

def get_ego_indices(natnet_ros_array, ego_view1, ego_view2, start_frame):
    ego_timestamps_1 = read_egocsv(ego_view1)
    ego_timestamps_2 = read_egocsv(ego_view2)
    
    ego1_corr_indices = np.asarray(loop_queryarray(natnet_ros_array, ego_timestamps_1))
    # print (ego1_corr_indices[:100], ego_timestamps_1[0], natnet_ros_array[0])
    ego1_corr_indices = ego1_corr_indices[:,1] - 210*np.ones(len(ego1_corr_indices))
    
    ego2_corr_indices = np.asarray(loop_queryarray(natnet_ros_array, ego_timestamps_2))
    ego2_corr_indices = ego2_corr_indices[:,1] - 210*np.ones(len(ego2_corr_indices))
    return ego1_corr_indices, ego2_corr_indices
    

def clip_video(input_video_file, ego_indices, exp_type, var, group, trial, zed):
    output_dir = "/scratch/msy9an/data/tro_data/{}/Variation_{}/Group_{}/Trial_{}".format(exp_type, var, group, trial)
    os.makedirs(output_dir, exist_ok=True)
    output_video_file = "{}/processed_ego_agent_{}.mp4".format(output_dir,zed)
    # if os.path.exists(output_video_file):
    #     return 
    print("output_video_file", output_video_file)
    cap = cv2.VideoCapture(input_video_file)
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    print ("fps ", fps)
    fps = 30
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # Specify the codec (codec options may vary)
    
    count = 0
    all_frames = list()
    print (int(ego_indices[-1]))
    while True:
        ret, frame = cap.read()
        if not ret:
            break  # End of video
        if count < int(ego_indices[-1])+10:
            all_frames.append(frame)
        count += 1
    cap.release()
    
    out = cv2.VideoWriter(output_video_file, fourcc, fps, (width, height))
    for count in (ego_indices):
        out.write(all_frames[int(count)])
        # print (frame.shape)
        
        # if count in ego_indices:
        #     out.write(frame)
        # count += 1
    
    out.release()
    cv2.destroyAllWindows()
    del all_frames
    return 

def getStartFrames(_file):
    start_frames = np.asarray(pd.read_csv(_file))
    return start_frames
    
def getFileName(experiment_type,variation_num,group_num,trial):
    skel_file =("/project/CollabRoboGroup/datasets/tro_data/{}/Variation_{}/Group_{}/Trial_{}/skeleton_data.csv".format(experiment_type,variation_num,group_num,trial))
    # rgb_file = ("/project/CollabRoboGroup/datasets/tro_data/{}/Variation_{}/Group_{}/Trial_{}/timestamp-withskeleton.csv".format(experiment_type,variation_num,group_num,trial)) #_hhc_{}_{}_{}.csv".format(experiment_type,variation_num,group_num,trial, variation_num,group_num,trial))
    rgb_file = ("/project/CollabRoboGroup/datasets/tro_data/{}/Variation_{}/Group_{}/Trial_{}/timestamp-withskeleton_hhc_{}_{}_{}.csv".format(experiment_type,variation_num,group_num,trial, variation_num,group_num,trial))
    ego_view1 = "/project/CollabRoboGroup/datasets/tro_pupil/{}-V{}-G{}-I{}-A2/world_timestamps.csv".format(experiment_type, variation_num, group_num, trial)
    ego_view2 = "/project/CollabRoboGroup/datasets/tro_pupil/{}-V{}-G{}-I{}-A3/world_timestamps.csv".format(experiment_type, variation_num, group_num, trial)
    
    return skel_file, rgb_file, ego_view1, ego_view2

parser = argparse.ArgumentParser()
parser.add_argument('experiment_type', type=str)
parser.add_argument('variation_num', type=str)
args = parser.parse_args()

experiment_type = args.experiment_type
variation_num = args.variation_num


_file = "/project/CollabRoboGroup/datasets/tro_data/{}_var{}.csv".format(experiment_type.lower(), variation_num)

start_frames = getStartFrames(_file)#[51:]
for i in range(len(start_frames)):
    print ((start_frames[i,2]))
    if start_frames[i,2] != ' ': # and not  math.isnan((start_frames[i,2])):
        group_num, trial_num, start_frame = int(start_frames[i,0]), int(start_frames[i,1]), int(start_frames[i,2])
        ego_files = "/project/CollabRoboGroup/datasets/tro_data/{}/Variation_{}/Group_{}/Trial_{}/zed*.mp4".format(experiment_type, variation_num, int(group_num), int(trial_num))
        natnet_ros_file, rgb_file, ego_view1, ego_view2 = getFileName("{}".format(experiment_type),"{}".format(variation_num),group_num,trial_num)
        if os.path.exists(ego_view1) and os.path.exists(ego_view2) and os.path.exists(natnet_ros_file):
            print (start_frame)
            try:
                natnet_ros_array = read_roscsv(natnet_ros_file)[start_frame:]
                        
                ego1_corr_indices, ego2_corr_indices = get_ego_indices(natnet_ros_array, ego_view1, ego_view2, start_frame)
                ego_rgb1 = ego_view1.replace("world_timestamps.csv","world-video.mp4")
                ego_rgb2 = ego_view2.replace("world_timestamps.csv","world-video.mp4")
                rgb_files = [ego_rgb1, ego_rgb2]
                
                ego_indicies = [(ego1_corr_indices), (ego2_corr_indices)]
                count = 0
                for rgb_file in sorted((rgb_files)):
                    print (rgb_file, start_frame, len(ego1_corr_indices))
                    clip_video(rgb_file, ego_indicies[count], experiment_type, variation_num, group_num, trial_num, count+2)
                    count += 1
            except:
                print ("errors for {} {}".format(group_num, trial_num))
                
                