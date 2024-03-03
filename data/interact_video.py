import os
import numpy as np
from tqdm import tqdm 
import h5py
import torch
import torch.utils.data as data
from sklearn.preprocessing import StandardScaler, MinMaxScaler
import matplotlib.pyplot as plt

class INTERACTDataset(data.Dataset):
    def __init__(self, config, split_name, data_aug=True, exp_type='HRC'):
        super(INTERACTDataset, self).__init__()
        self.exp_type = exp_type
        self._split_name = split_name
        self._data_aug = data_aug
        self.split = split_name
        self._interact_anno_dir = config.interact_anno_dir
        
        # self._interact_file_names = self._get_interact_names()
        self.interact_motion_input_length = config.motion.interact_input_length
        self.interact_motion_target_length = config.motion.interact_target_length_train   
        
        self.motion_dim = config.motion.dim
        self.unified_frames = self._load_summary(split_name)
        self._all_interact_motion_poses = self._load_all()
        
        self._all_video_data = self._load_all_video_chunks()
        # print ("self._all_interact_motion_poses ", np.asarray(self._all_interact_motion_poses).shape)
        self._file_length = len(self._all_interact_motion_poses)
        
    def _load_summary(self, split_name):
        _summary_file = "./data/{}_{}.csv".format(self.exp_type, split_name)
        
        import pandas as pd
        unified_frames = np.asarray(pd.read_csv(_summary_file, index_col=0))
        
        result_dict = {}
        for i in range(len(unified_frames)):
            result_dict['{}/frames'.format(unified_frames[i,0])]= '{}'.format(unified_frames[i,4])
        print (len(result_dict))
        return result_dict
            
    def __len__(self):
        if self._file_length is not None:
            return self._file_length
        return len(self._all_interact_motion_poses)
    
    def frame_preprocess(self, raw_skeleton_data):
        # print (np.mean(np.asarray(raw_skeleton_data[0]-raw_skeleton_data[-1])))
        return raw_skeleton_data #normalized_skeleton_data
        
    def _preprocess(self, interact_motion_feats):
        if interact_motion_feats is None:
            return None
        interact_seq_len = interact_motion_feats.shape[0]

        if self.interact_motion_input_length+self.interact_motion_target_length <= interact_seq_len:
            start =0 #np.random.randint(interact_seq_len - self.interact_motion_input_length - self.interact_motion_target_length + 1)
            end = start + self.interact_motion_input_length
        else:
            return None
        interact_motion_input = torch.zeros((self.interact_motion_input_length, interact_motion_feats.shape[1]))
        interact_motion_input[:end-start] = torch.FloatTensor(interact_motion_feats[start:end])
        
        interact_motion_target = torch.zeros((self.interact_motion_target_length, interact_motion_feats.shape[1]))
        interact_motion_target[:self.interact_motion_target_length] = torch.FloatTensor(interact_motion_feats[end:end+self.interact_motion_target_length])
        
        interact_motion = torch.cat([interact_motion_input, interact_motion_target], axis=0)
        # print ("interact_motion_input ", interact_motion_input.shape, interact_motion_target.shape)
        return interact_motion

    def processSkeletons_csv(self,skeleton_array, delim = 0):
        new_skeletonarray = skeleton_array
        # print (new_skeletonarray.shape)
        new_skeletonarray = new_skeletonarray[:,delim+0:delim+153]
        new_skeletonarray = new_skeletonarray.reshape(-1, 51, 3)
        new_skeletonarray = np.transpose(new_skeletonarray, (1,2,0))

        return new_skeletonarray

    
    def plot_test_skeletons(self, normalized_data):
        # joints = [1,2, 2,3, 3,4, 4,5, 4,10, 4,6, 6,7, 7,8, 8,9, 4,10, 10,11, 11,12, 12,13]
        path = './runs/skel'
        # path = "/project/CollabRoboGroup/datasets/tro_data/{}/Variation_{}/Group_{}/Trial_{}/Skeletons".format(experiment_type,variation_num,group_num,trial)
        joints = [1,2, 2,3, 3,4, 4,5, 4,10, 4,6, 6,7, 7,8, 8,9, 4,10, 10,11, 11,12, 12,13, 1,44, 44,45, 45,46,
        1,48, 48,49, 49,50, 50,51] #1,44, 44,45, 45,46,
        joints = [joint-1 for joint in joints]
        skeleton_array = np.asarray(normalized_data)
        skeleton_array_1 = self.processSkeletons_csv(skeleton_array)
        skeleton_array_2 = self.processSkeletons_csv(skeleton_array, delim=153)
        skeleton_array_3 = self.processSkeletons_csv(skeleton_array, delim=153*2)
        plt.ioff()
        fig, axes = plt.subplots(1,1)
        axes = fig.add_subplot(111, projection='3d')
        # if os.path.exists(path):
        #     return    
        
        for i in range(0, len(skeleton_array)):
            # ax1 = fig.add_subplot(111, projection='3d')
            axes.clear()
            axes.view_init(-90, 90)
            # axes.set_xlim(-2, 2)
            # axes.set_ylim(-2, 2)
            
            index_1 = i
            for index in range(0, len(joints), 2):
                gt_x = skeleton_array_1[joints[index:index+2], 0, i]
                gt_y = skeleton_array_1[joints[index:index+2], 1, i]
                gt_z = skeleton_array_1[joints[index:index+2], 2, i]
                axes.plot(gt_x, gt_y, gt_z, 'r', markersize=4, lw=2)
    
                gt_x1 = skeleton_array_2[joints[index:index+2], 0, i]
                gt_y1 = skeleton_array_2[joints[index:index+2], 1, i]
                gt_z1 = skeleton_array_2[joints[index:index+2], 2, i]
                axes.plot(gt_x1, gt_y1, gt_z1, 'g', markersize=4, lw=2)
    
                gt_x2 = skeleton_array_3[joints[index:index+2], 0, i]
                gt_y2 = skeleton_array_3[joints[index:index+2], 1, i]
                gt_z2 = skeleton_array_3[joints[index:index+2], 2, i]
                axes.plot(gt_x2, gt_y2, gt_z2, 'b', markersize=4, lw=2)
            plt.axis('off')
            plt.grid('off')
            plt.savefig("./logs/skel/agent1_skel_{}.png".format(index_1), bbox_inches='tight', pad_inches=0)
        plt.close(fig)
        
    def plot_2d(self, skeleton_array):
        positions_x = skeleton_array[:,0]
        positions_y = skeleton_array[:,1]
        
        time_steps = np.arange(0, len(positions_y))
        print (skeleton_array.shape)
        agent_2positions_x = skeleton_array[:,153]
        agent_2positions_y = skeleton_array[:,154]
        
        agent_3positions_x = skeleton_array[:,306]
        agent_3positions_y = skeleton_array[:,307]
        
        pos_x = [positions_x, agent_2positions_x, agent_3positions_x]
        pos_y = [positions_y, agent_2positions_y, agent_3positions_y]
        for i in range(3):
            fig = plt.figure(figsize=(10, 8))
            ax = fig.add_subplot(111, projection='3d')
            
            # Scatter plot with time as color
            sc = ax.scatter(pos_x[i], pos_y[i], time_steps, c=time_steps, cmap='viridis', marker='o')
            
            # Set labels and title
            ax.set_xlabel('X Position')
            ax.set_ylabel('Y Position')
            ax.set_zlabel('Time (s)')
            ax.set_title('3D Trajectory Plot with Time')
            
            # Add a colorbar for time
            cbar = fig.colorbar(sc)
            cbar.set_label('Time (s)')
            
            
    
            plt.savefig("/home/msy9an/TROExperiments/logs/2dPos/agent{}_skel.png".format(i), bbox_inches='tight', pad_inches=0)
            plt.close()
        
        
        
        print (positions_x.shape)
        
    def _load_all(self):
        zero_count = 0
        all_interact_motion_poses = list()
        with h5py.File('/scratch/msy9an/data/tro_video/skeleton_{}.h5'.format(self.split), 'r') as file:
            # Print the keys (group names) at the top level
            # Iterate through the keys and print the hierarchy
            def print_hierarchy(name, obj):
                if isinstance(obj, h5py.Dataset):
                    # This is an extra step for checking whether the corresponding video file exists
                    with h5py.File('/scratch/msy9an/data/tro_video/extracted_features_exo_2_{}.h5'.format(self.split), 'r') as video_file:
                        try:
                            video_name =  (self.unified_frames['/{}'.format(name)]).replace("zed_2", "zed_1")
                            video_data = video_file["{}/{}".format(video_name[1:],"extraced_features")]
                            
                            if video_data.shape[0] == obj[:].shape[0]-1:
                                motion_data = obj[:]
                                normalized_data = self.frame_preprocess(motion_data)
                                # self.plot_2d(normalized_data)
                                
                                # scaler = MinMaxScaler() #StandardScaler()#MinMaxScaler()
                                # # Fit the scaler to your motion data to compute mean and standard deviation
                                # scaler.fit(motion_data)
                                # # Transform the motion data to standardize it
                                # normalized_data = motion_data #scaler.transform(motion_data)
                                # self.plot_test_skeletons(normalized_data)
                                diff = normalized_data[0] - normalized_data[-1]
                                
                                truncated_slots = normalized_data.shape[0]//(self.interact_motion_input_length*2)
                                # print ("truncated_slots ", truncated_slots)
                                truncated_length = truncated_slots*self.interact_motion_input_length*2
                                normalized_data = normalized_data[:truncated_length].reshape(truncated_slots, self.interact_motion_input_length*2, normalized_data.shape[-1])
                                
                                
                                all_interact_motion_poses.extend(normalized_data)
                            else:
                                "nothing"
                                # print ("mismatch between skeleton and video")
                        except:
                            print ("skipping files ", name)
            file.visititems(print_hierarchy)
        # print (zero_count)
        npall_interact_motion_poses = np.array(all_interact_motion_poses).reshape(-1, all_interact_motion_poses[0].shape[1])
        print (npall_interact_motion_poses.shape)
        return all_interact_motion_poses


        
    def _load_all_video_chunks(self):
        all_interact_video_data = list()
        
        with h5py.File('/scratch/msy9an/data/tro_video/extracted_features_exo_2_{}.h5'.format(self.split), 'r') as file:
            # Print the keys (group names) at the top level
            # Iterate through the keys and print the hierarchy
            def print_hierarchy(name, obj):
                # print ("name ", name)
                if isinstance(obj, h5py.Dataset):
                    video_data = obj[:]
                    # print(video_data.shape)
                    truncated_slots = video_data.shape[0]//(self.interact_motion_input_length*2)
                    indices = [[name, index*self.interact_motion_input_length, (index+1)*self.interact_motion_input_length] for index in range(0, truncated_slots)]
                    assert len(indices) == truncated_slots
                    # print ("indices ", (indices), truncated_slots)
                    all_interact_video_data.extend(indices)
                            
            file.visititems(print_hierarchy)
            
        # print (len(all_interact_video_data))
        # all_interact_video_data = np.array(all_interact_video_data).reshape(-1, all_interact_video_data[0].shape[1])
        return all_interact_video_data
        
        
    def __getitem__(self, index):
        interact_motion_poses = self._all_interact_motion_poses[index]
        interact_motion = self._preprocess(interact_motion_poses)
        video_name, start, end = self._all_video_data[index][0], self._all_video_data[index][1], self._all_video_data[index][2]
        with h5py.File('/scratch/msy9an/data/tro_video/extracted_features_exo_2_{}.h5'.format(self.split), 'r') as file:
            # video_name = video_name.replace("frames", "extraced_features").replace("zed_1","zed_2")
            video_motion = file[video_name][start:end]
            # video_motion = np.transpose(file[video_name][start:end], (0, 3, 1, 2))
            file.close()    
            
        if interact_motion is None:
            print ("none")
            while interact_motion is None:
                index = np.random.randint(slef_file_length)
                interact_motion_poses = self._all_interact_motion_poses[index]
                interact_motion = self._preprocess(interact_motion_poses)
                
        if self._data_aug:
            if np.random.rand() > .5:
                idx = [i for i in range(interact_motion.size(0) - 1, -1, -1)]
                idx = torch.LongTensor(idx)
                interact_motion = interact_motion[idx]
        interact_motion_input = interact_motion[:self.interact_motion_input_length].float()
        interact_motion_target = interact_motion[self.interact_motion_input_length:].float()
        return interact_motion_input, interact_motion_target, video_motion
        
