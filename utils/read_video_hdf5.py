# import h5py

# # Open the HDF5 file
# with h5py.File('video_frames.h5', 'r') as file:
#     # Access a dataset by name
#     print("Top-level keys:", list(file.keys()))
#     dataset = file['project']
    

#     # Read the data from the dataset
#     data = dataset[:]
    
#     # Access attributes of the dataset
#     attributes = dataset.attrs
    
#     print (data.shape)
#     print (attributes)

# Data and attributes are now available for further processing
import h5py

# Open the HDF5 file in read mode
with h5py.File('/scratch/msy9an/data/tro_video/skeleton_frames.h5', 'r') as file:
    # Print the keys (group names) at the top level
    print("Top-level keys:", list(file.keys()))

    # Iterate through the keys and print the hierarchy
    def print_hierarchy(name, obj):
        # if isinstance(obj, h5py.Group):
        #     print("Group:", name)
        if isinstance(obj, h5py.Dataset):
            print("Dataset:", name, obj.shape)
    file.visititems(print_hierarchy)
