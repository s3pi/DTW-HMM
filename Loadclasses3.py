# -*- coding: utf-8 -*-
"""
Created on Tue Oct 30 18:33:32 2018

@author: uay_user
"""

import pandas as pd
import numpy as np
from os import listdir
from os.path import isfile, join

class MFCC_data:
    def load_mfcc_data_train():
        mfcc_data_folder = ["ti", "tI","TI"]
        mfcc_path = "/home/ganesan/PRAssignment3/Group02/Train"
        mfcc_allclass__allseq = []
        for clsname in mfcc_data_folder:
            path_mfcc_class = join(mfcc_path, clsname)
            print (path_mfcc_class)
            mfcc_class_allseq = []
            for mfcc_file in listdir(path_mfcc_class):
                path_mfcc_seque = join(path_mfcc_class,mfcc_file)
                data_1 = pd.read_csv(path_mfcc_seque, header = None, delimiter=' ')
                mfcc_d1 = np.array(data_1)
                mfcc_f1 = np.delete(mfcc_d1, 39, axis = 1)
                mfcc_class_allseq.append(mfcc_f1)
            mfcc_allclass__allseq.append(mfcc_class_allseq)
        return mfcc_allclass__allseq
    
    def load_mfcc_data_test():
        mfcc_data_folder = ["ti", "tI","TI"]
        mfcc_path = "/home/ganesan/PRAssignment3/Group02/Test"
        mfcc_allclass__allseq = []
        for clsname in mfcc_data_folder:
            path_mfcc_class = join(mfcc_path, clsname)
            print (path_mfcc_class)
            mfcc_class_allseq = []
            for mfcc_file in listdir(path_mfcc_class):
                path_mfcc_seque = join(path_mfcc_class,mfcc_file)
                data_1 = pd.read_csv(path_mfcc_seque, header = None, delimiter=' ')
                mfcc_d1 = np.array(data_1)
                mfcc_f1 = np.delete(mfcc_d1, 39, axis = 1)
                mfcc_class_allseq.append(mfcc_f1)
            mfcc_allclass__allseq.append(mfcc_class_allseq)
        return mfcc_allclass__allseq
    
    def load_mfcc_obs_seq_train(k):
        if(k == 8):
            mfcc_path = "/Users/3pi/Documents/Pattern Recognition/Ass3_DTW_HMM/Quantized_Data/Train/K8"
        if(k == 16):
            mfcc_path = "/Users/3pi/Documents/Pattern Recognition/Ass3_DTW_HMM/Quantized_Data/Train/K16"
        if(k == 32):
            mfcc_path = "/Users/3pi/Documents/Pattern Recognition/Ass3_DTW_HMM/Quantized_Data/Train/K32"
        mfcc_obs_allclass_allseq = []
        for cls_obs_seqs in listdir(mfcc_path):
            path_cls_obs_seqs_file = join(mfcc_path, cls_obs_seqs)
            obs_seqs_cls_i = []
            with open(path_cls_obs_seqs_file) as f1:
                for line in f1:
                    obs_line_array = [int(s) for s in line.split(' ')]
                    obs_seq_np = np.array(obs_line_array)
                    obs_seqs_cls_i.append(obs_seq_np)
            mfcc_obs_allclass_allseq.append(obs_seqs_cls_i) 
#            mfcc_obs = np.array(data_1)
#            mfcc_obs_allclass_allseq.append(mfcc_obs)
        return mfcc_obs_allclass_allseq
    
    def load_mfcc_obs_seq_test(k):
        if(k == 8):
            mfcc_path = "/Users/3pi/Documents/Pattern Recognition/Ass3_DTW_HMM/Quantized_Data/Test/K8"
        if(k == 16):
            mfcc_path = "/Users/3pi/Documents/Pattern Recognition/Ass3_DTW_HMM/Quantized_Data/Test/K16"
        if(k == 32):
            mfcc_path = "/Users/3pi/Documents/Pattern Recognition/Ass3_DTW_HMM/Quantized_Data/Test/K32"
            
        mfcc_obs_allclass_allseq = []
        for cls_obs_seqs in listdir(mfcc_path):
            path_cls_obs_seqs_file = join(mfcc_path, cls_obs_seqs)
            obs_seqs_cls_i = []
            with open(path_cls_obs_seqs_file) as f1:
                for line in f1:
                    obs_line_array = [int(s) for s in line.split(' ')]
                    obs_seq_np = np.array(obs_line_array)
                    obs_seqs_cls_i.append(obs_seq_np)
            mfcc_obs_allclass_allseq.append(obs_seqs_cls_i) 
#            mfcc_obs = np.array(data_1)
#            mfcc_obs_allclass_allseq.append(mfcc_obs)
        return mfcc_obs_allclass_allseq
        
MFCC_data.load_mfcc_obs_seq_train(8)