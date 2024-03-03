import pandas as pd
import numpy as np
import pickle

def read_sdkcsv(_file, delim='\t'):
    file_df = pd.read_csv(_file, header=None, delimiter=delim)
    # print (file_df)
    file_array = np.asarray(file_df)[:,0].reshape(-1,1)
    if int(file_array[0][0])/(10**18) >100:
        div = 100
    elif int(file_array[0][0])/(10**18) >10:
        div = 10    
    else:
        div = 1
    # div = 10 if int(file_array[0][0])/(10**18) >10 else 1
    # print ("div :", div)
    for i in range(file_array.shape[0]-2):
        try:
            file_array[i] = int(file_array[i][0])/div
        except:
            print (file_array[i])
            """print ("invalid literal")"""
    return file_array


def read_roscsv(_file):
    file_df = pd.read_csv(_file, header=None, index_col=0)
    # print (file_df)
    file_array = np.asarray(file_df)[:,0].reshape(-1,1)
    if int(file_array[0][0])/(10**18) >100:
        div = 100
    elif int(file_array[0][0])/(10**18) >10:
        div = 10    
    else:
        div = 1
    # print ("div :", div)
    for i in range(file_array.shape[0]-2):
        try:
            file_array[i] = int(file_array[i][0])/div
        except:
            print (file_array[i])
            """print ("invalid literal")"""

    return file_array

def read_egocsv(_file):
    file_df = pd.read_csv(_file, index_col=0)# ['timestamp [ns]']    
    # print (file_df)
    file_array = np.asarray(file_df)[:,1].reshape(-1,1)
    for i in range(file_array.shape[0]-2):
        try:
            file_array[i] = int(file_array[i][0])/1
        except:
            # print (file_array[i])
            print ("invalid literal")

    return file_array#[:-10]

def read_picklefile(_file):
    file = open(("skeleton{}.p".format("0")), 'rb')
    file_pickle = pickle.load(file)
    file_array = np.asarray(file_pickle).reshape(-1,1)
    for i in range(file_array.shape[0]):
        file_array[i] = int(str(file_array[i]).split("[")[-1].replace("]]", ""))
        file_array[i] = file_array[i][0]
    return file_array

def remove_value(_file_array, _index):
    truncated_array = np.delete(_file_array,_index)
    return truncated_array

def match_value(key, value_array):
    diff = abs(np.ones((len(value_array)-2, 1))*key[0] - value_array[:-2])

    min_index = np.argmin(diff)
    # print (diff[min_index], min_index)
    return min_index

def loop_queryarray(_query_array, value_array):
    corr_indices = []
    for i in range(_query_array.shape[0]):
        
        query =  _query_array[i]
        if type(query[0]) == np.int64:
            value_index =  match_value(query, value_array)
            truncated_value_array = remove_value(value_array, value_index)
            corr_indices.append([i, value_index])
    # for i in range(0, len(corr_indices), 150):
    #     print (corr_indices[i])
    return corr_indices
# natnet_ros_array = read_picklefile(natnet_ros_file)

# natnet_sdk_array = read_csvfile(natnet_sdk_file)
# natnet_ros_array = read_csvfile2(natnet_ros_file)
# corr_indices = loop_queryarray(natnet_ros_array, natnet_sdk_array)
# print (natnet_ros_array[0:100])
# print (natnet_sdk_array[0:100])
