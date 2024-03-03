import numpy as np
import pandas as pd
import pickle, os, glob, natsort, cv2, torch, random
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
import matplotlib.patches as mpatches


labels_file_loc = "/project/CollabRoboGroup/msy9an/GoalPrediction/src/utils/labels.csv"

labels = np.asarray(pd.read_csv(labels_file_loc))[:,-1:].reshape(-1,1)

def make_clusters(cluster_df, categorical_file_loc):

    clusters = cluster_df[:,2].reshape(-1,1)
    sequence_index = cluster_df[:,0].reshape(-1,1)
    cluster_num = categorical_file_loc.split("/")[-2].split("_")[-1]
    plthistogram(clusters, cluster_num)
    return
    labels = np.concatenate((sequence_index, labels), axis=1)

    cluster_len = dict()
    label_cluster = dict()
    cluster_label = dict()

    cluster_label = dict()
    for cluster_index in range(0, 27):
        bool, index =  np.where(labels[:,1].reshape(-1,1)==cluster_index)
        cluster_label[cluster_index] = labels[bool]

    #print (cluster_label)

    for label_index in range(0, 28):
        bool, index =  np.where(labels[:,2].reshape(-1,1)==label_index)
        label_cluster[label_index] = labels[bool]


    for label_index in range(1, 28):
        index, cluster, label = label_cluster[label_index][:,0], label_cluster[label_index][:,1], label_cluster[label_index][:,2]
        print (index, cluster)
        sequence_list = []
        file = open(("specific_index_{}.p".format(label_index)), 'wb')
        pickle.dump(index, file)
        file.close()
        file = open(("specific_clusters_{}.p".format(label_index)), 'wb')
        pickle.dump(cluster, file)
        file.close()

        """
        for i in range(1, index.shape[0]):
            if not index[i] - index[i-1] >=5:
                sequence_list.append([index[i-1], cluster[i-1]])
            else:
                sequence_array = np.asarray(sequence_list).reshape(-1,2)
                #plt.ylim(min(cluster), max(cluster))
                plt.step(sequence_array[:,0], sequence_array[:,1])
                plt.show()
                sequence_list = []

        break
        """

def plthistogram(cluster, cluster_num):
    bins_ = int(cluster_num.split("z")[0].replace("c", ""))
    plt.hist(cluster, bins=bins_,edgecolor='k', density=True)
    plt.xticks(range(bins_))
    plt.xlabel("action primitves")
    plt.ylabel("frequency")
    plt.savefig("/home/samin/Documents/hello_world/src/adversarialvae/log_dir_new/hist{}/gaussian_cluster_hist{}.pdf".format(cluster_num, cluster_num))
    print ("/home/samin/Documents/hello_world/src/adversarialvae/log_dir_new/gaussian_cluster_hist{}.pdf".format(cluster_num))
    plt.close()
def iterate_and_segment(cluster):
    """
    iterates images, segments them and then makes video
    """
    offset = 0

    base_path = os.path.join(os.getcwd(), "log_dir", "clusters", cluster)
    base_path = "/home/samin/Downloads/clusters/" + cluster
    class_path = os.path.join(os.getcwd(), "log_dir", "clusters", cluster, "*.png")
    class_path = "/home/samin/Downloads/clusters/" + cluster + "/*.png"
    temp_list = list()

    starting_index = int(natsort.natsorted(glob.glob(class_path))[0].split("/")[-1].split("_")[1+offset])
    #print (starting_index)

    for picture in natsort.natsorted(glob.glob(class_path)):
        class_file = picture.split("/")[-1]
        class_ = class_file.split("_")[0]
        iteration_num = int(class_file.split("_")[1+offset])
        segment_num = class_file.split("_")[2+offset].replace(".png","")
        print ("class_ {} iteration_num {} segment_num {}".format(class_, iteration_num, segment_num))
        if iteration_num - starting_index > 15:
            segment(temp_list, base_path)
            temp_list = list()
        else:
            temp_list.append(picture)
        starting_index = iteration_num

def segment(picture_list, base_path):
    """
    receives a list of image paths, makes video
    """
    offset = 0
    starting_segment_num = picture_list[0].split("_")[-1].replace(".png","")

    segment_list = list()
    for i, picture in enumerate(picture_list):
        class_file = picture.split("/")[-1]
        class_ = class_file.split("_")[0]
        iteration_num = int(class_file.split("_")[1 + offset])
        segment_num = class_file.split("_")[2+ offset].replace(".png","")
        print (segment_num, starting_segment_num)
        if segment_num != starting_segment_num:
            make_video(segment_list, starting_segment_num, iteration_num, base_path)
            segment_list = list()
        print (picture, segment_num)
        segment_list.append(cv2.imread(picture))
        starting_segment_num = segment_num

def make_video(segment_list, segment, iteration_num, base_path):
    """
    reads a list and creates a video for each segment
    """
    width, height, layers = segment_list[0].shape
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')

    output = "{}/{}_{}.mp4".format(base_path, iteration_num, segment)
    video_writer = cv2.VideoWriter(output, fourcc, 5, (height, width))

    for image in segment_list:
        video_writer.write(image)

    cv2.destroyAllWindows()
    video_writer.release()

def plotGaussian(gaussian_file_loc, categorical_file_loc, clusters):
    gaussian_df = np.asarray(pd.read_csv(gaussian_file_loc, index_col=0))

    cluster_num = categorical_file_loc.split("/")[-2].split("_")[-1]

    for i in range(gaussian_df.shape[0]):
        for j in range(gaussian_df.shape[1]):
            temp_string = str(gaussian_df[i][j]).split("(")[1]
            temp_string = str(temp_string).split(",")[0].replace(")", "")
            gaussian_df[i][j] = np.float64(temp_string)

    number_of_colors = 10
    #color = ["#"+''.join([random.choice('0123456789ABCDEF') for j in range(6)]) for i in range(number_of_colors)]
    color =['b', 'g', 'r', 'y'] #['#4DD5A5', '#81B405',
    print (color)

    patches = list()

    for i in range(len(color)):
        patches.append(mpatches.Patch(color=color[i], label= str(i)))

    """
    X_embedded = TSNE(n_components=2, perplexity=100).fit_transform(gaussian_df)
    file = open("X_embedded.p", 'wb')
    pickle.dump(X_embedded, file)
    file.close()

    """
    file = open(("X_embedded.p"), 'rb')
    X_embedded = pickle.load(file)
    file.close()

    fig = plt.figure()
    ax = fig.add_subplot()
    ax.set_xlim(-75,75)
    ax.set_ylim(-75,75)
    ax.set_xlabel("Dimension 1", fontsize=18)
    ax.set_ylabel("Dimension 2", fontsize=18)
    plt.xticks(fontsize=16)
    plt.yticks(fontsize=16)
    plt.tight_layout()

    selected_i = [0, 1, 2, 3]
    activties = ["RightBlock", "LeftBlock", "MiddleBlock", "Walk"]
    counter = 0
    for i in selected_i:#range(27):
        index, _ = np.where(labels == i)

        plotlabelwise(X_embedded[index], i, labels[index], color[counter], cluster_num, "label", ax, activties[counter])
        counter +=1
        #plotlabelandclusterwise(X_embedded[index], i, labels[index], clusters[index], color, cluster_num, "both")
    plt.legend(fontsize=16, loc='lower left')
    plt.savefig("/project/CollabRoboGroup/msy9an/GoalPrediction/outputs/csv_files/clusterall.png")
    plt.close()


def plotlabelwise(X_embedded_label, label, categorical, color, cluster_num, label_type, ax, activty):
    """
    fig = plt.figure()
    ax = fig.add_subplot()
    ax.set_xlim(-75,75)
    ax.set_ylim(-75,75)
    ax.set_xlabel("Dimension 1", fontsize=18)
    ax.set_ylabel("Dimension 2", fontsize=18)
    plt.xticks(fontsize=16)
    plt.yticks(fontsize=16)
    plt.tight_layout()
    """

    cluster_dict = dict()
    counter = 0
    for i in range(categorical.shape[0]):
        if categorical[i] not in cluster_dict.values():
            cluster_dict[str(counter)] = categorical[i]
            counter += 1
    number_of_colors = counter


    for i in range(X_embedded_label.shape[0]):
        key = get_key(categorical[i], cluster_dict)
        ax.scatter(X_embedded_label[i,0], X_embedded_label[i,1], c=color)

    ax.scatter(X_embedded_label[i-1,0], X_embedded_label[i-1,1], c=color, label=activty)
    plt.legend(fontsize=16, loc='lower left')

    #plt.savefig("/project/CollabRoboGroup/msy9an/GoalPrediction/outputs/csv_files/cluster_labels{}.png".format(label))

def plotlabelandclusterwise(X_embedded_label, label, categorical_label, categorical_cluster, color, cluster_num, label_type):
    """
    First calls plotlablewise, then calls plot clusterwise
    """


    cluster_label = dict()
    cluster_cluster = dict()
    counter = 0
    for i in range(categorical_label.shape[0]):
        if categorical_label[i] not in cluster_label.values():
            cluster_label[str(counter)] = categorical_label[i]


    for i in range(categorical_cluster.shape[0]):
        if categorical_cluster[i] not in cluster_cluster.values():
            cluster_cluster[str(counter)] = categorical_cluster[i]


    for i in range(X_embedded_label.shape[0]):
        key = get_key(categorical_label[i], cluster_label)
        for j in range(10):
            index, _ = np.where(categorical_cluster == j)
            if len(index)>3:

                fig = plt.figure()
                ax = fig.add_subplot()
                ax.set_xlim(-100,100)
                ax.set_ylim(-100,100)

                ax.scatter(X_embedded_label[index,0], X_embedded_label[index,1], c=color[int(key)])

                plt.savefig("/home/samin/Documents/hello_world/src/adversarialvae/log_dir_new/hist{}/gaussian_{}_{}_{}.png".format(cluster_num, label_type, label, j))
                plt.close()
                #plt.show()


def get_key(val, my_dict):
    for key, value in my_dict.items():
         if val == value:
             return key


def iteratehist():

    
    categorical_file_loc = "/project/CollabRoboGroup/msy9an/GoalPrediction/outputs/csv_files/test_cat_25.csv"
    gaussian_file_loc = "/project/CollabRoboGroup/msy9an/GoalPrediction/outputs/csv_files/test_gauss_25.csv"
    cluster_num = categorical_file_loc.split("/")[-2].split("_")[-1]

    cluster_df = np.asarray(pd.read_csv(categorical_file_loc))
    clusters = cluster_df[:,1].reshape(-1,1)
    #make_clusters(cluster_df, categorical_file_loc)
    plotGaussian(gaussian_file_loc, categorical_file_loc, clusters)

iteratehist()



"""
for i in range (1, 5):
    iterate_and_segment(str(i))
"""
