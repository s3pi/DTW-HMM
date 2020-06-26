from copy import deepcopy
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
from matplotlib import pyplot as plt
import math
import pylab as plt
from os import listdir
from os.path import isfile, join
from Loadclasses3 import MFCC_data
from kmeans_GMM import find_performance_matrix as fp
from kmeans_GMM import find_accuracy as fa

def main():
	k = 16
	N = 3
	allclass_allseq_train = MFCC_data.load_mfcc_obs_seq_train(k)
	allclass_allseq_test = MFCC_data.load_mfcc_obs_seq_test(k)
	print(len(allclass_allseq_test[0]), len(allclass_allseq_test[1]), len(allclass_allseq_test[2]))
	pi_final_all_classes = []
	A_final_all_classes = []
	B_final_all_classes = []
	for class_index in range(len(allclass_allseq_train)):
		print("\nClass number", class_index)
		each_class_all_obs_seq = allclass_allseq_train[class_index]
		pi_final_each_class, A_final_each_class, B_final_each_class = train_HMM_each_class(each_class_all_obs_seq, N, k, class_index)
		pi_final_all_classes.append(pi_final_each_class)
		A_final_all_classes.append(A_final_each_class)
		B_final_all_classes.append(B_final_each_class)

	confusion_matrix = compute_confusion_matrix(allclass_allseq_test, pi_final_all_classes, A_final_all_classes, B_final_all_classes, N)
	print("\nconfusion_matrix")
	print_matrix(confusion_matrix)

	performance_matrix = fp(confusion_matrix)
	print("\nperformance_matrix")
	print_matrix(performance_matrix)

	class_accuracy = fa(confusion_matrix)
	print("\nclass accuracy: ", class_accuracy)

def compute_confusion_matrix(allclass_allseq_test, pi_final_all_classes, A_final_all_classes, B_final_all_classes, N):
    confusion_matrix = [[0, 0, 0], [0, 0, 0], [0, 0, 0]]
    for class_index in range(3):
        for seq_index in range(len(allclass_allseq_test[class_index])):
            class_num = test_which_class(allclass_allseq_test[class_index][seq_index], pi_final_all_classes, A_final_all_classes, B_final_all_classes, N)
            confusion_matrix[class_index][class_num] += 1

    return confusion_matrix

def test_which_class(obs_seq, pi_final_all_classes, A_final_all_classes, B_final_all_classes, N):
	prob_obs_seq_each_class = []
	for class_index in range(3):
		alpha_mat_per_seq, prob_of_alpha_mat_per_seq = compute_alpha_mat_each_seq(pi_final_all_classes[class_index], A_final_all_classes[class_index], B_final_all_classes[class_index], obs_seq, N)
		prob_obs_seq_each_class.append(prob_of_alpha_mat_per_seq)

	return prob_obs_seq_each_class.index(max(prob_obs_seq_each_class))

def train_HMM_each_class(each_class_all_obs_seq, N, k, class_index):
	A_avg_each_class, B_avg_each_class = compute_A_B_avg(each_class_all_obs_seq, N, k)
	prob_obs_given_hmm_model_old = 0
	prob_obs_given_hmm_model_new = 0
	alpha_mat_each_class_all_seq = []
	beta_mat_each_class_all_seq = []
	prob_each_class_all_iters = [0]
	pi_mat = [1] + ([0] * (N-1))
	for obs_seq_index in range(len(each_class_all_obs_seq)):
		alpha_mat_per_seq, prob_of_alpha_mat_per_seq = compute_alpha_mat_each_seq(pi_mat, A_avg_each_class, B_avg_each_class, each_class_all_obs_seq[obs_seq_index], N)
		alpha_mat_each_class_all_seq.append(alpha_mat_per_seq)
		beta_mat_per_seq, prob_of_beta_mat_per_seq = compute_beta_mat_each_seq(pi_mat, A_avg_each_class, B_avg_each_class, each_class_all_obs_seq[obs_seq_index], N)
		beta_mat_each_class_all_seq.append(beta_mat_per_seq)
		prob_obs_given_hmm_model_new += np.log(prob_of_alpha_mat_per_seq)
	
	prob_each_class_all_iters.append(prob_obs_given_hmm_model_new)
	pi_avg_restimated_each_class, A_avg_restimated_each_class, B_avg_restimated_each_class = baum_welch_EM_method(each_class_all_obs_seq, alpha_mat_each_class_all_seq, beta_mat_each_class_all_seq, A_avg_each_class, B_avg_each_class, N, k)
	# print(prob_obs_given_hmm_model_new)
	while abs(prob_obs_given_hmm_model_old - prob_obs_given_hmm_model_new) > 0.1:
		prob_obs_given_hmm_model_old = prob_obs_given_hmm_model_new
		prob_obs_given_hmm_model_new = 0
		for obs_seq_index in range(len(each_class_all_obs_seq)):
			alpha_mat_per_seq, prob_of_alpha_mat_per_seq = compute_alpha_mat_each_seq(pi_avg_restimated_each_class, A_avg_restimated_each_class, B_avg_restimated_each_class, each_class_all_obs_seq[obs_seq_index], N)
			alpha_mat_each_class_all_seq.append(alpha_mat_per_seq)
			beta_mat_per_seq, prob_of_beta_mat_per_seq = compute_beta_mat_each_seq(pi_avg_restimated_each_class, A_avg_restimated_each_class, B_avg_restimated_each_class, each_class_all_obs_seq[obs_seq_index], N)
			beta_mat_each_class_all_seq.append(beta_mat_per_seq)
			prob_obs_given_hmm_model_new += np.log(prob_of_alpha_mat_per_seq)

		prob_each_class_all_iters.append(prob_obs_given_hmm_model_new)
		# print(prob_obs_given_hmm_model_new)
		if abs(prob_obs_given_hmm_model_old - prob_obs_given_hmm_model_new) > 0.1:
			pi_avg_restimated_each_class, A_avg_restimated_each_class, B_avg_restimated_each_class = baum_welch_EM_method(each_class_all_obs_seq, alpha_mat_each_class_all_seq, beta_mat_each_class_all_seq, A_avg_each_class, B_avg_each_class, N, k)
	
	print(prob_each_class_all_iters)
	plot_iters_vs_loglikelihood(prob_each_class_all_iters, class_index, N, k)

	return pi_avg_restimated_each_class, A_avg_restimated_each_class, B_avg_restimated_each_class

def plot_iters_vs_loglikelihood(prob_each_class_all_iters, class_index, N, k):
    plt.plot(prob_each_class_all_iters[1:])
    if class_index == 0:
        title = "Class 'small_t_small_i'"
    elif class_index == 1:
        title = "Class 'small_t_cap_I'"
    elif class_index == 2:
        title = "Class 'cap_T_cap_I'"
    plt.title(str(title))
    plt.xlabel('Iterations')
    plt.ylabel('log likelihood')
    plt.savefig(str(title) + "_" + "N" + str(N) + "_K" + str(k) + ".png")
    plt.clf()

def print_matrix(matrix):
    for i in range(len(matrix)):
        print(matrix[i])

def baum_welch_EM_method(each_class_all_obs_seq, alpha_mat_each_class_all_seq, beta_mat_each_class_all_seq, A_avg_each_class, B_avg_each_class, N, M):
	A_avg_restimated_each_class = np.zeros((N, N))
	B_avg_restimated_each_class = np.zeros((N, M))
	eff_no_transitions_from_1st_state_all_seq = np.zeros((N))
	for obs_seq_index in range(len(each_class_all_obs_seq)):
		#Expectation-Step
		zeta_mat_per_seq = compute_zeta_mat_each_seq(each_class_all_obs_seq[obs_seq_index], alpha_mat_each_class_all_seq[obs_seq_index], beta_mat_each_class_all_seq[obs_seq_index], A_avg_each_class, B_avg_each_class, N)
		gamma_mat_per_seq = compute_gamma_mat_each_seq(each_class_all_obs_seq[obs_seq_index], alpha_mat_each_class_all_seq[obs_seq_index], beta_mat_each_class_all_seq[obs_seq_index], N)
		#Maximization-Step
		eff_no_transitions_from_1st_state_all_seq += gamma_mat_per_seq[1]
		A_avg_restimated_each_class += (np.sum(zeta_mat_per_seq, axis = 0) / np.sum(gamma_mat_per_seq[1:], axis = 0))
		B_avg_restimated_each_class +=  compute_B_restimated_per_seq(each_class_all_obs_seq[obs_seq_index], gamma_mat_per_seq, N, M)

	pi_avg_restimated_each_class = eff_no_transitions_from_1st_state_all_seq / len(each_class_all_obs_seq)
	A_avg_restimated_each_class = A_avg_restimated_each_class / len(each_class_all_obs_seq)
	B_avg_restimated_each_class = B_avg_restimated_each_class / len(each_class_all_obs_seq)

	return pi_avg_restimated_each_class, A_avg_restimated_each_class, B_avg_restimated_each_class

def compute_B_restimated_per_seq(obs_seq, gamma_mat_per_seq, N, M):
	B_restimated_per_seq = np.zeros((N, M))
	T = len(obs_seq)
	for j in range(N):
		for k in range(M):
			numerator = 0
			for t in range(T):
				if obs_seq[t] == k:
					numerator += gamma_mat_per_seq[t][j]
			B_restimated_per_seq[j][k] = numerator / np.sum(gamma_mat_per_seq[:, j])

	return B_restimated_per_seq

def compute_gamma_mat_each_seq(obs_seq, alpha_mat, beta_mat, N):
	T = len(obs_seq)
	gamma_mat = np.zeros((T, N))
	for t in range(T):
		denominator = 0
		for i in range(N):
			denominator += (alpha_mat[t][i] * beta_mat[t][i])
		for i in range(N):
			gamma_mat[t][i] = (alpha_mat[t][i] * beta_mat[t][i]) / denominator

	return gamma_mat

def compute_zeta_mat_each_seq(obs_seq, alpha_mat, beta_mat, A_avg_each_class, B_avg_each_class, N):
    T = len(obs_seq)
    zeta_mat = np.zeros((T, N, N))
    for t in range(T-1):    # here our loop starts from 0, so T-1
        denominator = 0
        for i in range(N):
            for j in range(N):  
                denominator += alpha_mat[t][i] * A_avg_each_class[i][j] * B_avg_each_class[j][obs_seq[t+1]] * beta_mat[t+1][j]
        for i in range(N):
            for j in range(N):
                numerator = alpha_mat[t][i] * A_avg_each_class[i][j] * B_avg_each_class[j][obs_seq[t+1]] * beta_mat[t+1][j]
                zeta_mat[t][i][j] = numerator / denominator
             
    return zeta_mat

def compute_A_B_avg(each_class_all_obs_seq, N, k):
	A_avg_each_class = np.zeros((N, N))
	B_avg_each_class = np.zeros((N, k))
	for obs_seq in each_class_all_obs_seq:
		state_seq = compute_state_sequence(obs_seq, N)
		A_avg_each_class += find_A_each_seq(N, state_seq)
		B_avg_each_class += find_B_each_seq(N, k, obs_seq, state_seq)
	A_avg_each_class /= len(each_class_all_obs_seq)
	B_avg_each_class /= len(each_class_all_obs_seq)

	return A_avg_each_class, B_avg_each_class

def compute_state_sequence(obs_seq, N):
	min_no_sym_each_state = len(obs_seq) // N
	state_seq = [(i // min_no_sym_each_state) for i in range(min_no_sym_each_state * N)]
	if len(obs_seq) % N > 0:
		state_seq += [N - 1] * (len(obs_seq) % N)
	return state_seq

def find_A_each_seq(N, state_seq):
	min_no_sym_each_state = len(state_seq) // N
	A = np.zeros((N, N))
	for i in range(N-1):
		A[i][i] = (min_no_sym_each_state - 1) / min_no_sym_each_state
		A[i][i+1] = (1 / min_no_sym_each_state)

	A[N-1][N-1] = 1

	return A

def find_B_each_seq(N, k, obs_seq, state_seq):
	B = np.zeros((N, k))
	for i in range(len(state_seq)):
		B[state_seq[i]][obs_seq[i]] += 1
	no_times_in_each_state = np.sum(B, axis = 1)
	for i in range(len(no_times_in_each_state)):
		for j in range(k):
			B[i][j] /= no_times_in_each_state[i]

	return B


def compute_alpha_mat_each_seq(pi_mat, A_avg_each_class, B_avg_each_class, obs_seq, N):
	T = len(obs_seq)
	alpha_mat = np.zeros((T, N))
	
	#Initialization
	for j in range(N):
		alpha_mat[0][j] = pi_mat[j] * B_avg_each_class[j][obs_seq[0]]

	#Induction
	for t in range(1, T):
		for j in range(N):
			temp = 0
			for i in range(N):
				temp += (alpha_mat[t-1][i] * A_avg_each_class[i][j])
			alpha_mat[t][j] = temp * B_avg_each_class[j][obs_seq[t]]
	
	prob_of_alpha_mat_per_seq = np.sum(alpha_mat[-1], axis = 0)

	return alpha_mat, prob_of_alpha_mat_per_seq

def compute_beta_mat_each_seq(pi_mat, A_avg_each_class, B_avg_each_class, obs_seq, N):
    T = len(obs_seq)
    beta_mat = np.zeros((T, N))
    beta_mat[T-1: ] = 1
    for t in range(T-2, -1, -1):        #as array goes from 0 to T-1, T-2 for T-1 here..
        for i in range(N):
            temp = 0
            for j in range(N):
                temp += (A_avg_each_class[i][j] * B_avg_each_class[j][obs_seq[t+1]] * beta_mat[t+1][j])
            beta_mat[t][i] = temp
    
    prob_of_beta_mat_per_seq = 0
    for i in range(N):
        prob_of_beta_mat_per_seq += pi_mat[i] * B_avg_each_class[i][obs_seq[0]] * beta_mat[0][i]

    return beta_mat, prob_of_beta_mat_per_seq

def compute_zeta_mat_each_seq(obs_seq, alpha, beta, A_matrix_one_class_avg, B_matrix_one_class_avg, N):
    T = len(obs_seq)
    zeta_mat = np.zeros((T, N, N))
    for t in range(T-1):    # here our loop starts from 0, so T-1
        denominator = 0
        for i in range(N):
            for j in range(N):  
                denominator += alpha[t][i] * A_matrix_one_class_avg[i][j] * B_matrix_one_class_avg[j][obs_seq[t+1]] * beta[t+1][j]
        for i in range(N):
            for j in range(N):
                numer = alpha[t][i] * A_matrix_one_class_avg[i][j] * B_matrix_one_class_avg[j][obs_seq[t+1]] * beta[t+1][j]
                zeta_val = numer / denominator
                zeta_mat[t][i][j] = zeta_val
    return zeta_mat
    
def build_zeta(kth_mfcc_obs_allclass_allseq_train,Alpha_matrix_all_class_each_matrix,Beta_matrix_all_class_each_matrix,A_matrix_all_class_avg, B_matrix_all_class_avg, N):
    zeta_matrix_all_class_each_matrix = []
    for clsno in range(len(Alpha_matrix_all_class_each_matrix)):
        zeta_matrix_per_class = []
        for seqs in range(len(Alpha_matrix_all_class_each_matrix[clsno])):
                        
#            seqs_xi = kth_mfcc_obs_allclass_allseq_train[clsno][seqs]
            alpha = Alpha_matrix_all_class_each_matrix[clsno][seqs]
            beta = Beta_matrix_all_class_each_matrix[clsno][seqs]
            obs_seq = kth_mfcc_obs_allclass_allseq_train[clsno][seqs]
            zeta = compute_zeta_mat_each_seq(obs_seq, alpha, beta, A_matrix_all_class_avg[clsno], B_matrix_all_class_avg[clsno], N)
            zeta_matrix_per_class.append(zeta)
        zeta_matrix_all_class_each_matrix.append(zeta_matrix_per_class)
    return zeta_matrix_all_class_each_matrix

main()