import os
import cv2
import h5py
import numpy as np
import  pandas as pd
import argparse
from models.transformer.vision_model import SwinTransformer

def preprocess_video(video_path, feature_extractor, target_shape=(224, 224)):
    cap = cv2.VideoCapture(video_path)
    frames = []
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frame = cv2.resize(frame, target_shape)
        frames.append(frame)
    cap.release()
    print ("frames ", np.asarray(frames).shape)
    features = feature_extractor(frames.cuda())
    
    return features #np.array(frames)
    
def preprocess_skeleton(video_path):
    frames = pd.read_csv(video_path, index_col=0)
    
    return np.array(frames)#[2:]


parser = argparse.ArgumentParser()
parser.add_argument('key', type=str)
parser.add_argument('split', type=str)
args = parser.parse_args()

key = args.key
_file = "{}_file_1".format(key)
_frames = "{}_frames_1".format(key)
_split = args.split


data_dir = "path_to_video_folder"
hdf5_file = "/scratch/msy9an/data/tro_video/extracted_features_{}_2_{}.h5".format(key, _split)
video_files = pd.read_csv('/project/CollabRoboGroup/msy9an/TROExperiments/src/data/{}.csv'.format(_split))[_file]
rgb_frames = pd.read_csv('/project/CollabRoboGroup/msy9an/TROExperiments/src/data/{}.csv'.format(_split))[_frames]


feature_extractor = SwinTransformer().cuda()
    
print (video_files)
# Create an HDF5 file
with h5py.File(hdf5_file, "w") as f:
    for index, video_file in enumerate(video_files):
        if rgb_frames[index] > 0:
            video_path = video_file #os.path.join(data_dir, video_file)
            frames = preprocess_video(video_path, feature_extractor)
            print (frames.shape)
            # Create a group for each video
            group = f.create_group(video_file)
            
            # Store frames in the group
            group.create_dataset("extraced_features", data=frames)
