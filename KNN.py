# -*- coding: utf-8 -*-
"""
Created on Tue Oct 30 18:29:31 2018

@author: PR_Group02_Ganesan_Preethi
"""

import numpy as np
import pandas as pd
from Loadclasses3 import MFCC_data
from Performance_matrix import find_performance_matrix

    
def Consonant_vowel_data():
        data_1 = pd.read_csv("/home/ganesan/PRAssignment3/Group02/Train/ti/01_ti_70_5.mfcc", header = None, delimiter=' ')
        data_2 = pd.read_csv("/home/ganesan/PRAssignment3/Group02/Train/ti/01_ti_84_3.mfcc", header = None, delimiter=' ') 
        mfcc_d1 = np.array(data_1)
        mfcc_f1 = np.delete(mfcc_d1, 39, axis = 1)
        mfcc_d2 = np.array(data_2)
        mfcc_f2 = np.delete(mfcc_d2, 39, axis = 1)
        print (mfcc_f1.shape)
        print("Mfcc")
        return mfcc_f1, mfcc_f2

def euclid_distance(mfcc_f1, mfcc_f2):
    euclid_matrix = np.zeros((mfcc_f1.shape[0],mfcc_f2.shape[0]))
#    distances = np.zeros((1,mfcc_f2.shape[0]))
    for i in range(mfcc_f1.shape[0]):
        for j in range(mfcc_f2.shape[0]):
            distance = np.linalg.norm(mfcc_f1[i] - mfcc_f2[j],axis = 0)
            euclid_matrix[i][j] = distance
    return euclid_matrix

def find_dtw_distance(euclid_matrix):
    i,j = euclid_matrix.shape
    dtw_matrix = np.zeros((i,j))
    dtw_matrix[0][0] = euclid_matrix[0][0]
    for jj in range(1, j):          # finding dtw of first rows
        dtw_matrix[0][jj] = euclid_matrix[0][jj] + dtw_matrix[0][jj-1]
    for ii in range(1, i):   # finding dtw of first column
        dtw_matrix[ii][0] = euclid_matrix[ii][0] + dtw_matrix[ii-1][0]
    for ii in range(1, i):
        for jj in range(1, j):
            dtw_matrix[ii][jj] = euclid_matrix[ii][jj] + min(dtw_matrix[ii-1][jj-1], dtw_matrix[ii-1][jj], dtw_matrix[ii][jj-1])
    dtw_dist = dtw_matrix[i-1][j-1] / (i*j)         #though dtw_matrix is i-1, j-1 for the last row, the dtw distance denominator is i*j, because we consider the sequence length, not index
    return dtw_dist

def compute_KNN(N_class, mfcc_test_allclass__allseq, mfcc_train_allclass__allseq, K): 
    confusion_matrix = [[0, 0, 0], [0, 0, 0], [0, 0, 0]]
    for ni in range(N_class):
        for seqs in range(len(mfcc_test_allclass__allseq[ni])):
            test_seq = mfcc_test_allclass__allseq[ni][seqs]
#            dtw_dist_testseq_with_all_refseq = np.zeros((N_class, len(mfcc_train_allclass__allseq[ni])))
            dtw_dist_testseq_with_all_refseq = []
            for ni_ref in range(N_class):
                size_refseq_class_i = len(mfcc_train_allclass__allseq[ni_ref])
                dtw_dist_testseq_with_class_i_refseq = np.zeros((size_refseq_class_i))
                for seqs_ref in range(size_refseq_class_i):
                    ref_seq = mfcc_train_allclass__allseq[ni_ref][seqs_ref]
                    euclid_matrix = euclid_distance(test_seq, ref_seq)
                    dtw_dist = find_dtw_distance(euclid_matrix)
                    dtw_dist_testseq_with_class_i_refseq[seqs_ref] = dtw_dist
                dtw_dist_testseq_with_all_refseq.append(np.sort(dtw_dist_testseq_with_class_i_refseq, axis = 0, kind = 'quicksort'))
            dtwdist_first_k, dtwdist_class_name = [], []
            for ii in range(N_class):
                for jj in range(K):
                    dtwdist_first_k.append(dtw_dist_testseq_with_all_refseq[ii][jj])
                    dtwdist_class_name.append(ii)
            dtwdist_first_k_np = np.array(dtwdist_first_k)
            dtwdist_class_name_np = np.array(dtwdist_class_name)
            perm_arr = dtwdist_first_k_np.argsort()
            dtwdist_class_name_np_sorted = dtwdist_class_name_np[perm_arr]
            dtwdist_class_name_sorted_K = dtwdist_class_name_np_sorted[:K]
            counts = np.bincount(dtwdist_class_name_sorted_K)
            whclass = np.argmax(counts)
            confusion_matrix[ni][whclass] += 1
    return confusion_matrix

def main():
    mfcc_train_allclass__allseq = MFCC_data.load_mfcc_data_train()
    mfcc_test_allclass__allseq = MFCC_data.load_mfcc_data_test()
    N_class = len(mfcc_test_allclass__allseq)
#    mfcc_f1, mfcc_f2 = Consonant_vowel_data()
    knn_kval = [4, 8, 16, 32] 
    for k in knn_kval:
       Kval = k
       confusion_matrix = compute_KNN(N_class, mfcc_test_allclass__allseq, mfcc_train_allclass__allseq, Kval)
       print(confusion_matrix)
#    dtw_matrix = np.zeros((mfcc_f1.shape[0],mfcc_f2.shape[0]))
#    euclid_matrix = euclid_distance(mfcc_f1, mfcc_f2)
#    dtw_matrix, dtw_dist = find_dtw_distance(euclid_matrix)
#    print (dtw_dist)

main()
    