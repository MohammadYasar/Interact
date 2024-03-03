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




def clip_video(input_video_file, start_frame, experiment_type, var, group, trial, zed):
    output_dir = "/scratch/msy9an/data/tro_data/{}/Variation_{}/Group_{}/Trial_{}".format(experiment_type, var, group, trial)
    os.makedirs(output_dir, exist_ok=True)
    output_video_file = "{}/processed_exo_zed_{}.mp4".format(output_dir,zed)
    if os.path.exists(output_video_file):
        return 
    print("output_video_file", output_video_file)
    cap = cv2.VideoCapture(input_video_file)
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # Specify the codec (codec options may vary)
    out = cv2.VideoWriter(output_video_file, fourcc, fps, (width, height))
    all_frames = list()
    sync_frames = list()
    count = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break  # End of video
        # print (frame.shape)
        if count>= start_frame:
            out.write(frame)
        count += 1
    cap.release()
    out.release()
    cv2.destroyAllWindows()
    
    return 

def getStartFrames(_file):
    start_frames = np.asarray(pd.read_csv(_file))
    return start_frames
    
def getFileName(experiment_type,variation_num,group_num,trial):
    skel_file =("/project/CollabRoboGroup/datasets/tro_data/{}/Variation_{}/Group_{}/Trial_{}/skeleton_data.csv".format(experiment_type,variation_num,group_num,trial))
    # rgb_file = ("/project/CollabRoboGroup/datasets/tro_data/{}/Variation_{}/Group_{}/Trial_{}/timestamp-withskeleton.csv".format(experiment_type,variation_num,group_num,trial)) #_hhc_{}_{}_{}.csv".format(experiment_type,variation_num,group_num,trial, variation_num,group_num,trial))
    rgb_file = ("/project/CollabRoboGroup/datasets/tro_data/{}/Variation_{}/Group_{}/Trial_{}/timestamp-withskeleton_hhc_{}_{}_{}.csv".format(experiment_type,variation_num,group_num,trial, variation_num,group_num,trial))
    ego_view1 = "/project/CollabRoboGroup/datasets/tro_pupil/HHC-V{}-G{}-I{}-A2/world_timestamps.csv".format(variation_num, group_num, trial)
    ego_view2 = "/project/CollabRoboGroup/datasets/tro_pupil/HHC-V{}-G{}-I{}-A3/world_timestamps.csv".format(variation_num, group_num, trial)
    
    return skel_file, rgb_file, ego_view1, ego_view2


parser = argparse.ArgumentParser()
parser.add_argument('experiment_type', type=str)
parser.add_argument('variation_num', type=str)
args = parser.parse_args()

experiment_type = args.experiment_type
variation_num = args.variation_num



count = 0    
_file = "/project/CollabRoboGroup/datasets/tro_data/{}_var{}.csv".format(experiment_type.lower(), variation_num)
start_frames = getStartFrames(_file)
print (int(start_frames[0,2]))
for i in range(len(start_frames)):
    print (("start_frames ", start_frames[i,2]))
    if start_frames[i,2] != ' ':# and not  math.isnan((start_frames[i,2])):
        group_num, trial_num, start_frame = int(start_frames[i,0]), int(start_frames[i,1]), int(start_frames[i,2])
        rgb_files = "/project/CollabRoboGroup/datasets/tro_data/{}/Variation_{}/Group_{}/Trial_{}/zed*.mp4".format(experiment_type, variation_num, int(group_num), int(trial_num))
        print (rgb_files)
        count = 1
        for rgb_file in sorted(glob.glob(rgb_files)):
            print (rgb_file, start_frame)
            clip_video(rgb_file, start_frame, experiment_type, "{}".format(variation_num), group_num, trial_num, count)
            count += 1
