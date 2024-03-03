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

# Demo Instruction: python3 plot_skelton.py hhc 2 2 1



length = 51

def processSkeletons(skeleton_array):
    new_skeletonarray = np.array(skeleton_array).reshape(skeleton_array.shape[0]//43, 43, 3)
    new_skeletonarray = new_skeletonarray[:,0:13, :]
    new_skeletonarray = new_skeletonarray.reshape(-1, 13, 3)
    new_skeletonarray = np.transpose(new_skeletonarray, (1,2,0))

    return new_skeletonarray

def processSkeletons_csv(skeleton_array, delim = 0):
    new_skeletonarray = np.array(skeleton_array)
    print (new_skeletonarray.shape)
    new_skeletonarray = new_skeletonarray[:,delim+0:delim+153]
    new_skeletonarray = new_skeletonarray.reshape(-1, 51, 3)
    new_skeletonarray = np.transpose(new_skeletonarray, (1,2,0))

    return new_skeletonarray



def plot_skeletons(corr_indices, natnet_sdk_file, experiment_type,subject_num,trial):
    # joints = [1,2, 2,3, 3,4, 4,5, 4,10, 4,6, 6,7, 7,8, 8,9, 4,10, 10,11, 11,12, 12,13]
    path = "/project/CollabRoboGroup/datasets/tro_data/{}/Variation_{}/Group_{}/Trial_{}/Skeletons".format(experiment_type,variation_num,group_num,trial)
    joints = [1,2, 2,3, 3,4, 4,5, 4,10, 4,6, 6,7, 7,8, 8,9, 4,10, 10,11, 11,12, 12,13, 1,44, 44,45, 45,46,
    1,48, 48,49, 49,50, 50,51] #1,44, 44,45, 45,46,
    joints = [joint-1 for joint in joints]
    df = pd.read_csv(natnet_sdk_file, index_col=0, header=None, delimiter="\t")
    skeleton_array = np.asarray(df)
    skeleton_array_1 = processSkeletons_csv(skeleton_array)
    skeleton_array_2 = processSkeletons_csv(skeleton_array, delim=153)
    skeleton_array_3 = processSkeletons_csv(skeleton_array, delim=153*2)
    plt.ioff()
    fig, axes = plt.subplots(1,1)
    axes = fig.add_subplot(111, projection='3d')
    if os.path.exists(path):
        return    
    os.makedirs(path)
    
    for i in range(0, len(corr_indices)):
        # ax1 = fig.add_subplot(111, projection='3d')
        axes.clear()
        axes.view_init(-90, 90)
        axes.set_xlim(-2, 2)
        axes.set_ylim(-2, 2)
        
        index_1 = i
        for index in range(0, len(joints), 2):
            gt_x = skeleton_array_1[joints[index:index+2], 0, corr_indices[index_1][-1]]
            gt_y = skeleton_array_1[joints[index:index+2], 1, corr_indices[index_1][-1]]
            gt_z = skeleton_array_1[joints[index:index+2], 2, corr_indices[index_1][-1]]
            axes.plot(gt_x, gt_y, gt_z, 'r', markersize=4, lw=2)

            gt_x1 = skeleton_array_2[joints[index:index+2], 0, corr_indices[index_1][-1]]
            gt_y1 = skeleton_array_2[joints[index:index+2], 1, corr_indices[index_1][-1]]
            gt_z1 = skeleton_array_2[joints[index:index+2], 2, corr_indices[index_1][-1]]
            axes.plot(gt_x1, gt_y1, gt_z1, 'g', markersize=4, lw=2)

            gt_x2 = skeleton_array_3[joints[index:index+2], 0, corr_indices[index_1][-1]]
            gt_y2 = skeleton_array_3[joints[index:index+2], 1, corr_indices[index_1][-1]]
            gt_z2 = skeleton_array_3[joints[index:index+2], 2, corr_indices[index_1][-1]]
            axes.plot(gt_x2, gt_y2, gt_z2, 'b', markersize=4, lw=2)
        plt.axis('off')
        plt.grid('off')
        plt.savefig("/project/CollabRoboGroup/datasets/tro_data/{}/Variation_{}/Group_{}/Trial_{}/Skeletons/agent1_skel_{}.png".format(experiment_type,variation_num,group_num,trial, index_1), bbox_inches='tight', pad_inches=0)

    plt.close(fig)

def make_video(experiment_type, subject_num, trial):
    count = 1
    for i in range(0, 1800*2, 150):
        filename = "{}/{}/{}/skel_video_hhc_{}.avi" .format(experiment_type, subject_num, trial, count)
        image = cv2.imread("{}/{}/{}/skel_video_hhc_{}.avi" .format(experiment_type, subject_num, trial, i))
        w, h, _ = image.shape
        video_writer = cv2.VideoWriter(filename, cv2.VideoWriter_fourcc(*'XVID'), 30, (h,w))

        for j in range(i, i+150):
            image = cv2.imread("test/0/3/skel_{}.png".format(j))
            video_writer.write(image)
            print (j)

        cv2.destroyAllWindows()
        video_writer.release()
        count +=1

def write_egovideo(input_video_file, ego_indices, var, group, trial, agent):
    output_dir = "/scratch/msy9an/data/tro_ego/Variation_{}/Group_{}/Trial_{}".format(var, group, trial)
    os.makedirs(output_dir, exist_ok=True)
    output_video_file = "{}/processed_ego_agent_{}.mp4".format(output_dir,agent)
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
        if count in ego_indices:
            out.write(frame)
        count += 1
    cap.release()
    out.release()
    cv2.destroyAllWindows()
    
    return 

def getFileName(experiment_type,variation_num,group_num,trial):
    skel_file =("/project/CollabRoboGroup/datasets/tro_data/{}/Variation_{}/Group_{}/Trial_{}/skeleton_data.csv".format(experiment_type,variation_num,group_num,trial))
    # rgb_file = ("/project/CollabRoboGroup/datasets/tro_data/{}/Variation_{}/Group_{}/Trial_{}/timestamp-withskeleton.csv".format(experiment_type,variation_num,group_num,trial)) #_hhc_{}_{}_{}.csv".format(experiment_type,variation_num,group_num,trial, variation_num,group_num,trial))
    rgb_file = ("/project/CollabRoboGroup/datasets/tro_data/{}/Variation_{}/Group_{}/Trial_{}/timestamp-withskeleton_hhc_{}_{}_{}.csv".format(experiment_type,variation_num,group_num,trial, variation_num,group_num,trial))
    ego_view1 = "/project/CollabRoboGroup/datasets/tro_pupil/HHC-V{}-G{}-I{}-A2/world_timestamps.csv".format(variation_num, group_num, trial)
    ego_view2 = "/project/CollabRoboGroup/datasets/tro_pupil/HHC-V{}-G{}-I{}-A3/world_timestamps.csv".format(variation_num, group_num, trial)
    
    return skel_file, rgb_file, ego_view1, ego_view2

def check_timings(corresponding_inidices, natnet_ros_array, timestamp_array):
    count = 0
    for i in range(1, len(corresponding_inidices)):
        diff_timestamp = natnet_ros_array[corresponding_inidices[i][0]]-timestamp_array[corresponding_inidices[i][1]]
        diff_window = diff_timestamp/10e7 - 1.0/60
        if (diff_window>0):
            count +=1
    print ("length of array {} count of positives {} percentage {}".format(len(corresponding_inidices), count, float(count)/len(corresponding_inidices)))

def writeSkeleton(corr_indices, natnet_sdk_file, experiment_type,subject_num,trial):
    path = "/project/CollabRoboGroup/datasets/tro_data/{}/Variation_{}/Group_{}/Trial_{}/Skeletons".format(experiment_type,variation_num,group_num,trial)
    df = pd.read_csv(natnet_sdk_file, index_col=0, header=None, delimiter="\t")
    skeleton_array = np.asarray(df)
    new_skeleton_array = list()
    for i in range(0, len(corr_indices)):
        new_skeleton_array.append(skeleton_array[corr_indices[i][-1]])
    new_skeleton_array = np.asarray(new_skeleton_array)
    new_skeleton_array = pd.DataFrame(new_skeleton_array)
    # print ("/project/CollabRoboGroup/datasets/tro_data/{}/Variation_{}/Group_{}/Trial_{}/processed_skeleton.csv".format(experiment_type,variation_num,group_num,trial))
    new_skeleton_array.to_csv("/project/CollabRoboGroup/datasets/tro_data/{}/Variation_{}/Group_{}/Trial_{}/processed_skeleton.csv".format(experiment_type,variation_num,group_num,trial))

def get_ego_indices(natnet_ros_array, ego_view1, ego_view2):
    ego_timestamps_1 = read_egocsv(ego_view1)
    ego_timestamps_2 = read_egocsv(ego_view2)
    
    ego1_corr_indices = np.asarray(loop_queryarray(natnet_ros_array, ego_timestamps_1))
    ego1_corr_indices = ego1_corr_indices[:,1] - 100*np.ones(len(ego1_corr_indices))
    
    ego2_corr_indices = np.asarray(loop_queryarray(natnet_ros_array, ego_timestamps_2))
    ego2_corr_indices = ego2_corr_indices[:,1] - 100*np.ones(len(ego2_corr_indices))
    return ego1_corr_indices, ego2_corr_indices
    
count = 0    
for experiment_type in (["HHC"]):
    for variation_num in range(2,3):
        for group_num in range(1, 22):
            for trial in range(1, 4): 
                try:
                    print(experiment_type,variation_num,group_num,trial)
                    skel_file, rgb_file, ego_view1, ego_view2 = getFileName(experiment_type,variation_num,group_num,trial)
                    
                    if os.path.exists(skel_file) and os.path.exists(rgb_file):
                        print ("entering")
                        if os.path.exists("/project/CollabRoboGroup/datasets/tro_data/{}/Variation_{}/Group_{}/Trial_{}/processed_skeleton.csv".format(experiment_type,variation_num,group_num,trial)):
                            print ("Already written ", experiment_type,variation_num,group_num,trial)
                            count += 1
                            # continue 
                        
                        print ("writing")
                        natnet_ros_file = skel_file 
                        natnet_sdk_file = rgb_file  
                        natnet_sdk_array = read_sdkcsv(natnet_sdk_file, delim='\t')
                        natnet_ros_array = read_roscsv(natnet_ros_file)
                        # corr_indices = loop_queryarray(natnet_ros_array, natnet_sdk_array)
                        # writeSkeleton(corr_indices, natnet_sdk_file, experiment_type, group_num, trial)
                        ego1_corr_indices, ego2_corr_indices = get_ego_indices(natnet_ros_array, ego_view1, ego_view2)
                        
                        if len(ego1_corr_indices)>5:
                            ego_rgb1 = ego_view1.replace("world_timestamps.csv","world-video.mp4")
                            write_egovideo(ego_rgb1, ego1_corr_indices, variation_num, group_num, trial, 2)
                            ego_rgb2 = ego_view2.replace("world_timestamps.csv","world-video.mp4")
                            write_egovideo(ego_rgb2, ego2_corr_indices, variation_num, group_num, trial, 3)

                        # plot_skeletons(corr_indices, natnet_sdk_file, experiment_type, group_num, trial)
                except:
                    print ("exception code ", rgb_file)
print ("count ", count)                    