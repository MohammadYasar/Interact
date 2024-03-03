import os, sys, cv2
import glob
import csv
import pandas as pd

variations = ['Variation_1', 'Variation_2']
trials = ['Trial_1', 'Trial_2', 'Trial_3']
exp_types = ['HHC', 'HRC']

skeleton_file = ""
train_data_dict = dict()
test_data_dict = dict()
keys = ["skeleton_file", "skeleton_frames", "exo_file_1", "exo_frames_1", "exo_file_2", "exo_frames_2", "ego_file_2", "ego_frames_2", "ego_file_3", "ego_frames_3"]

for key in keys:
    train_data_dict[key] = list()
    test_data_dict[key] = list()

def readSkeleton(_filelist):
    row_count = 0
    _file = None
    print ("file ", _filelist)
    if len(_filelist)>=1:
        _file = _filelist[0]
        print ("file ", _file)
    if _file:
        if os.path.exists(_file):
            with open(_file, 'r') as file:
                csv_reader = csv.reader(file)
                row_count = sum(1 for row in csv_reader)

    return row_count, _file

def readZed(_filelist):
    # Reading rgb files
    _file = None
    total_frames = 0
    if len(_filelist)>=1:
        _file = _filelist[0]
        # print ("file ", _file)
    if _file:
        # data_dict[key].append(_file)
        if os.path.exists(_file):
            cap = cv2.VideoCapture(_file)
    
            # Get the total number of frames in the video
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            
            # Release the video capture object
            cap.release()
            # print ("total_frames: ", total_frames)
    return total_frames, _file
    
for exp_type in exp_types:
    for var in variations:
        for group in range(1, 22):
            data_dir = "/scratch/msy9an/data/tro_data"
            group_dir =  "{}/{}/{}/Group_{}/*".format(data_dir, exp_type, var, group)
            print (group_dir)
            data_dict = train_data_dict if group%2 == 0 else test_data_dict
            for trial in sorted(glob.glob(group_dir)):
                data_files = "{}/*".format(trial)
                # _files in sorted(glob.glob(data_files))
                zed_file_1 = glob.glob("{}/processed_exo_zed_1*.mp4".format(trial))
                
                zed_file_2 = glob.glob("{}/processed_exo_zed_2*.mp4".format(trial))
                ego_file_2 = glob.glob("{}/processed_ego_agent_2*".format(trial))
                ego_file_3 = glob.glob("{}/processed_ego_agent_3*".format(trial))
                skel_file =  glob.glob("{}/new_processed_skeleton*".format(trial))              
                print (skel_file)
                    
                        
                # processed_skeleton.csv".format(data_dir, exp_type, var, group, trial)
                # zed_file_1 = glob.glob("/project/CollabRoboGroup/datasets/tro_data/{}/{}/Group_{}/{}/zed_1*.mp4".format(exp_type, var, group, trial))
                # zed_file_2 = glob.glob("/project/CollabRoboGroup/datasets/tro_data/{}/{}/Group_{}/{}/zed_2*.mp4".format(exp_type, var, group, trial))
                
                # zed_file_1 = zed_file_1[0] if len(zed_file_1)>=1 else zed_file_1
                # zed_file_2 = zed_file_2[0] if len(zed_file_2)>=1 else zed_file_2
                
                # print ("zed_file_1 ", zed_file_1)
                # # Reading skeleton files
                total_frames, _file = readSkeleton(skel_file)
                data_dict["skeleton_file"].append(_file)
                data_dict["skeleton_frames"].append(total_frames)
                
                total_frames, _file = readZed(zed_file_1)
                data_dict["exo_file_1"].append(_file)
                data_dict["exo_frames_1"].append(total_frames)

                total_frames, _file = readZed(zed_file_2)
                data_dict["exo_file_2"].append(_file)
                data_dict["exo_frames_2"].append(total_frames)
                

                total_frames, _file = readZed(ego_file_2)
                data_dict["ego_file_2"].append(_file)
                data_dict["ego_frames_2"].append(total_frames)
                
                total_frames, _file = readZed(ego_file_3)
                data_dict["ego_file_3"].append(_file)
                data_dict["ego_frames_3"].append(total_frames)
               
        
train_file = "/home/msy9an/TROExperiments/data/corrected_train.csv"
test_file = "/home/msy9an/TROExperiments/data/corrected_test.csv"

for key in keys:
    # print (key)
    print (len(data_dict[key]))
df = pd.DataFrame.from_dict(train_data_dict)
df.to_csv(train_file, header=keys, index=False)

df = pd.DataFrame.from_dict(test_data_dict)
df.to_csv(test_file, header=keys, index=False)

print(f"CSV file '{train_file}' created.")                
print(f"CSV file '{test_file}' created.")        
        
    
