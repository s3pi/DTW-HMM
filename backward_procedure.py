#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Nov  3 12:44:20 2018

@author: ganesan
"""

import numpy as np
import pandas as pd
from Loadclasses3 import MFCC_data


def apply_state_splitting_of_sequence(seqs_xi, N):
    div = len(seqs_xi) / N
    state_for_seqs_xi = []
    j = 0
    for i in range(len(seqs_xi)):
       state_for_seqs_xi.append(int(i/div))
       
    return state_for_seqs_xi

def find_state_transition_per_seqs(state_for_seqs_xi):
    n = 1+ max(state_for_seqs_xi) #number of states
    M = [[0]*n for i in range(n)]
    for (i,j) in zip(state_for_seqs_xi,state_for_seqs_xi[1:]):
        M[i][j] += 1
    for row in M:
        s = sum(row)
        if s > 0:
            row[:] = [f/s for f in row]
    st_prob_M = np.array(M)
    return st_prob_M

def find_state_observation_per_seqs(seqs_xi, state_for_seqs_xi, N, k):
    obs_prob_M = np.zeros((N, k))
    for l in range(len(seqs_xi)):
        i = seqs_xi[l]
        j = state_for_seqs_xi[l]
        obs_prob_M[j][i] +=1
        
    for row in obs_prob_M:
        s = sum(row)
        if s > 0:
            row[:] = [f/s for f in row]
    return obs_prob_M

    
def find_avg_A_Matrix(A_matrix_per_class_np, N):
    A_matrix = np.mean(A_matrix_per_class_np,axis = 0)
    return A_matrix

def find_avg_B_Matrix(B_matrix_per_class_np, N):
    B_matrix = np.mean(B_matrix_per_class_np,axis = 0)
    return B_matrix

    
def find_state_transition_matrix(kth_mfcc_obs_allclass_allseq_train, N):
    A_matrix_all_class = []
    A_matrix_all_class_each_matrix = []
    for clsno in range(len(kth_mfcc_obs_allclass_allseq_train)):
        A_matrix_per_class = []
        for seqs in range(len(kth_mfcc_obs_allclass_allseq_train[clsno])):
            seqs_xi = kth_mfcc_obs_allclass_allseq_train[clsno][seqs]
            state_for_seqs_xi = apply_state_splitting_of_sequence(seqs_xi, N)
            A_matrix_per_seq = find_state_transition_per_seqs(state_for_seqs_xi)
            A_matrix_per_class.append(A_matrix_per_seq)
        A_matrix_per_class_np = np.array(A_matrix_per_class)            # for N number of seqs
        A_matrix_class = find_avg_A_Matrix(A_matrix_per_class_np, N)
        print(A_matrix_class)
        A_matrix_all_class.append(A_matrix_class)
        A_matrix_all_class_each_matrix.append(A_matrix_per_class_np)

    return A_matrix_all_class, A_matrix_all_class_each_matrix


def find_state_observation_matrix(kth_mfcc_obs_allclass_allseq_train, N, k):
    B_matrix_all_class = []
    B_matrix_all_class_each_matrix = []
    for clsno in range(len(kth_mfcc_obs_allclass_allseq_train)):
        B_matrix_per_class = []
        for seqs in range(len(kth_mfcc_obs_allclass_allseq_train[clsno])):
            seqs_xi = kth_mfcc_obs_allclass_allseq_train[clsno][seqs]
            state_for_seqs_xi = apply_state_splitting_of_sequence(seqs_xi, N)
            B_matrix_per_seq = find_state_observation_per_seqs(seqs_xi, np.array(state_for_seqs_xi), N, k)
            B_matrix_per_class.append(B_matrix_per_seq)
        B_matrix_per_class_np = np.array(B_matrix_per_class)            # for N number of seqs
        B_matrix_class = find_avg_B_Matrix(B_matrix_per_class_np, N)
        B_matrix_all_class.append(B_matrix_class)
        B_matrix_all_class_each_matrix.append(B_matrix_per_class_np)
    return B_matrix_all_class, B_matrix_all_class_each_matrix

def compute_pi_matrix(N):
    pi_arr = np.zeros((N))
    pi_arr[0] = 1
    return pi_arr

def build_beta_matrix_perseqs(seqs_xi,A_matrix_all_class_avg, B_matrix_all_class_avg, pi_arr, N, k):
    T = len(seqs_xi)
    beta = np.zeros((T,N))
    beta[T-1:] = 1
    for tt in range(T-2, -1, -1):        #as array goes from 0 to T-1, T-2 for T-1 here..
        for i in range(N):
            val = 0
            for j in range(N):
                val += (A_matrix_all_class_avg[i][j] * B_matrix_all_class_avg[j][seqs_xi[tt+1]] * beta[tt+1][j])
            beta[tt][i] = val
    prob = 0
    for i in range(N):
        prob += pi_arr[i] * B_matrix_all_class_avg[i][0] * beta[0][i]
        
    return beta, prob


def backward_method(kth_mfcc_obs_allclass_allseq_train,A_matrix_all_class_avg, B_matrix_all_class_avg, pi_arr, N, k):
#    initialize_matrix T*N... and do further compute Beta by today.. 
#    write the outer loop for baum-walch method... 
    Beta_matrix_all_class = []
    Beta_matrix_all_class_each_matrix = []
    prob_o_given_lam_all_class = []
    for clsno in range(len(kth_mfcc_obs_allclass_allseq_train)):
        Beta_matrix_per_class = []
        prob_o_given_lam_per_class = []
        for seqs in range(len(kth_mfcc_obs_allclass_allseq_train[clsno])):
#            beta_i = np.zeros((k, N))
            seqs_xi = kth_mfcc_obs_allclass_allseq_train[clsno][seqs]
            beta, prob = build_beta_matrix_perseqs(seqs_xi, A_matrix_all_class_avg[clsno], B_matrix_all_class_avg[clsno], pi_arr, N, k)
            Beta_matrix_per_class.append(beta)
            prob_o_given_lam_per_class.append(prob)
#            Beta_matrix_per_class_np = np.array(Beta_matrix_per_class)
        Beta_matrix_all_class_each_matrix.append(Beta_matrix_per_class)
        prob_o_given_lam_all_class.append(prob_o_given_lam_per_class)
    return Beta_matrix_all_class_each_matrix

def main():
    k = 8
    N = 4
    kth_mfcc_obs_allclass_allseq_train = MFCC_data.load_mfcc_obs_seq_train(k)
    # kth_mfcc_obs_allclass_allseq_test = MFCC_data.load_mfcc_obs_seq_test(k)
    A_matrix_all_class_avg, A_matrix_all_class_each_matrix = find_state_transition_matrix(kth_mfcc_obs_allclass_allseq_train, N)
    # B_matrix_all_class_avg, B_matrix_all_class_each_matrix = find_state_observation_matrix(kth_mfcc_obs_allclass_allseq_train, N, k)
    # pi_arr = compute_pi_matrix(N)
    # Beta_matrix_all_class_each_matrix = backward_method(kth_mfcc_obs_allclass_allseq_train,A_matrix_all_class_avg, B_matrix_all_class_avg, pi_arr, N, k)
    # print("working")
    
main()