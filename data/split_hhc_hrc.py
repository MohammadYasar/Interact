import numpy as np
import pandas as pd
import argparse


# split = "HHC"
# remove = "HRC"
# _input_file = "corrected_test.csv"
# _output_file = "{}_test.csv".format(split)

# df = pd.read_csv(_input_file)
# df_clean = df.dropna()
# print ("length of df before removing rows ", len(df))
# rows_to_remove = list()

# for index, row in df.iterrows():
#     if type(row['skeleton_file']) != float:
#         if remove in row['skeleton_file']:
#             print (row['skeleton_file'])
#             rows_to_remove.append(index)
#     else:
#             rows_to_remove.append(index)
# new_df = df.drop(rows_to_remove)
# print ("length of df after removing rows ", len(new_df))

# new_df.to_csv(_output_file)


hhc_train = "HHC_train.csv"
hhc_test = "HHC_test.csv"

hrc_train = "HRC_train.csv"
hrc_test = "HRC_test.csv"

train_files = [hhc_train, hhc_test]
test_files = [hrc_train, hrc_test]

hhc_df_train = pd.read_csv(hhc_train, index_col=0)
hhc_df_test = pd.read_csv(hhc_test, index_col=0)

all_train_df = pd.concat([hhc_df_train, hhc_df_test])

hrc_df_train = pd.read_csv(hrc_train, index_col=0)
hrc_df_test = pd.read_csv(hrc_test, index_col=0)

all_test_df = pd.concat([hrc_df_train, hrc_df_test])

all_train_df.to_csv('HHC_HRC_train.csv')
all_test_df.to_csv('HHC_HRC_test.csv')

