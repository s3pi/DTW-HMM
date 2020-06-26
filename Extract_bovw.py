# -*- coding: utf-8 -*-
"""
Created on Wed Oct 17 23:59:48 2018

@author: uay_user
"""
import numpy as np
from os import listdir
from os.path import isfile, join
import cv2
import pandas as pd
from copy import deepcopy


def assigncenterBovw():
        # Converged center after k-means
    path = "/home/uay_user/PRAssignment2/BovW_cluster_centers.txt"
    with open(path) as infile:
        clst_center = np.fromstring(infile.read().replace("[","").replace("]", ""), sep="   ").reshape(32,24)
        return clst_center

def compute_BoVW(clst_cent, k):
    img_features_Folder = ["Train_Feature_Vectors", "Test_Feature_Vectors"]
    Feature_Vectors_Folder = ["Train_BovW_Feature_Vectors", "Test_BoVW_Feature_Vectors"]
    dir_path = "/home/uay_user/PRAssignment2/SceneImage_features/"
    
    for i in range(2):
        train_folder = ["coast", "industrial_area", "pagoda"]
        for scene_type in train_folder:
            all_images_path = join(join(dir_path, img_features_Folder[i]), scene_type)
            all_txt_files_path = join(dir_path, Feature_Vectors_Folder[i], scene_type)
            for each_fetr_file_name in listdir(all_images_path):
                fetr_path = join(all_images_path, each_fetr_file_name)
                #print(img_path)
                bovw_feature_vector = [0] * k
                cols = (0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23)
                fetr1 = pd.read_csv(fetr_path, header = None, delimiter=',', usecols = cols)
                fetr_in_np_form = np.array(fetr1)
                whc = assigncenter(fetr_in_np_form, k, clst_cent)
                print (len(whc))
                sum1 = 0
                for j in range(len(whc)):
                    bovw_feature_vector[whc[j]] +=1
                    sum1 += whc[j]
                print (bovw_feature_vector)
                print (sum1)
                if not os.path.exists(all_txt_files_path):
                    os.makedirs(all_txt_files_path)
                file_name, ext = each_fetr_file_name.split(".")
                txt_file_path = join(all_txt_files_path, file_name + ".txt")
                
                ptrf = open(txt_file_path, "w+")
                ab = str(bovw_feature_vector).strip('[').strip(']')
                ptrf.write(ab + "\n")

def assigncenter(fetr_in_np_form, k, clst_cent):
        # Converged center after k-means
       
#        print (center)
#        print (center[0])
        curr_centers = deepcopy(clst_cent)  
        n, d = fetr_in_np_form.shape
        distances = np.zeros((n, k))
        for i in range(k):
            distances[:,i] = np.linalg.norm(fetr_in_np_form - curr_centers[i], axis=1)  
#        for i in range(k):
#            for j in range(n):
#                distances[j][i] = np.linalg.norm(fetr_in_np_form[j] - clst_cent[i], axis =1)
        whcluster = np.argmin(distances, axis=1)
#        print (whcluster)
#        cluster_data = []
#        cluster_data = np.array([testclsf[whcluster==i] for i in range(k)])
        return whcluster
    
def main():
    clst_center = assigncenterBovw()
    k = 32
    compute_BoVW(clst_center, k)
    print (clst_center)
    
main()
