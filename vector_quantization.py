import numpy as np
from os import listdir
from os.path import isfile, join
from Loadclasses3 import MFCC_data
from copy import deepcopy

def main():
    all_train_data = MFCC_data.load_mfcc_data_test()
    all_train_data_as_one_list = club_all_files_all_classes_train(all_train_data)
    K = 8
    init_centers = find_initial_random_centers(all_train_data_as_one_list, K)
    centers_after_kmeans, whcluster = computeKMC(all_train_data_as_one_list, init_centers, K)
    cluster_center = "kmeans_centers_for" + str(K) +".npy"
    np.save(cluster_center,centers_after_kmeans)
    print(centers_after_kmeans)
    vector_quantization(all_train_data, centers_after_kmeans, K)

#def vector_quantization(all_train_data, whcluster):
#    path = "/Users/3pi/Documents/Pattern Recognition/Ass3_DTW_HMM/Train"
#    folders_in_train = listdir("/Users/3pi/Documents/Pattern Recognition/Ass3_DTW_HMM/Train")
#    for file_name in folders_in_train:
#        txt_file_path = join(path, file_name + ".txt")
#        for 
#        quantized_file = 
#        f = open(txt_file_path, "w+")
#        for each in feature_vectors:
#            f.write(str(each) + "\n")
#        f.close()
def vector_quantization(all_train_data, centers, K):
    cls1_path = "/home/ganesan/PRAssignment3/MFCC_data/cls1_ti_seqs.txt"
    cls2_path = "/home/ganesan/PRAssignment3/MFCC_data/cls2_tI_seqs.txt"
    cls3_path = "/home/ganesan/PRAssignment3/MFCC_data/cls3_TI_seqs.txt"
    write_files = [cls1_path, cls2_path, cls3_path]
    for clsno in range(len(all_train_data)):
        fptr_path_to_write = open(write_files[clsno], "w+")
        for seqs in range(len(all_train_data[clsno])):
            seqs_xi = all_train_data[clsno][seqs]
            whc_np = assigncenter(seqs_xi, centers, K)
#            whc_np_str = str(whc_np).strip('[').strip(']')
            fptr_path_to_write.write(join(" ".join(map(str, whc_np))))
            fptr_path_to_write.write("\n")
            print ("here")
            
def assigncenter(seqs_xi, clst_cent, k):
    curr_centers = deepcopy(clst_cent)  
    n, d = seqs_xi.shape
    distances = np.zeros((n, k))
    for i in range(k):
        distances[:,i] = np.linalg.norm(seqs_xi - curr_centers[i], axis=1)  
    whcluster = np.argmin(distances, axis=1)
    return whcluster            
              
    

def find_initial_random_centers(data, k):
    n = data.shape[0] # Number of training data points
    c = data.shape[1] # Number of features in the data (number of dimentions)
    mean = np.mean(data, axis = 0) #39 dimention mean
    std = np.std(data, axis = 0) #39 dimention std
    init_centers = []
    for i in range(k):
        init_centers = np.append(init_centers, data[np.random.randint(n)], axis = 0)
    init_centers = np.resize(init_centers, (k, 39))

    return init_centers

def computeKMC(data, init_centers, k):   
    curr_centers = deepcopy(init_centers) 
    n = data.shape[0]   #number of rows (data points)
    prev_centers = np.zeros(curr_centers.shape)  # to store old centers
    distances = np.zeros((n, k))      
    error = np.linalg.norm(curr_centers - prev_centers) #sum of distances from all (curr_centers, prev_centers) is expected to be zero.   
    itr = 0
    while error >= 0.001:
        # Euclidean distance between each data point and each centers. 
        # distance[] is a (num of data points * num of clusters) matrix.
        itr = itr + 1
        for i in range(k):
            distances[:,i] = np.linalg.norm(data - curr_centers[i], axis=1) 
        whcluster = np.argmin(distances, axis=1) #whcluster contains the information about which cluster, each data point belongs to.
        prev_centers = deepcopy(curr_centers)
        #MStep Finding the new cluster center with new points..
        for i in range(k):
            if(np.any(whcluster == i)):         # If there is no data associated with cluster, the center remain the same..
                curr_centers[i] = np.mean(data[whcluster == i], axis=0)
            else:
                print("no cluster associated in iteration number", itr)
                continue;
#        curr_centers = curr_centers[~np.isnan(curr_centers)]
        error = np.linalg.norm(curr_centers - prev_centers)

    return curr_centers, whcluster

def club_all_files_all_classes_train(all_train_data):#class --> file --> seq --> 39d data
    all_train_data_as_one_list = []
    for class_no in range(len(all_train_data)):
        for file_no in range(len(all_train_data[class_no])):
            for seq_no in range(len(all_train_data[class_no][file_no])):
                all_train_data_as_one_list.append(all_train_data[class_no][file_no][seq_no])

    all_train_data_as_one_list = np.array(all_train_data_as_one_list)
    return all_train_data_as_one_list


main()