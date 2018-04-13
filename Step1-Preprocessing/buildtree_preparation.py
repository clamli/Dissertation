import sys
import numpy as np
import scipy.sparse
from scipy.sparse import csr_matrix, find
import read_write as rw
import matlab.engine

'''
finput_iu_rating_matrix_train = "Data/iu_sparse_matrix_train.npz"
finput_title_sim_matrix = "Data/title_similarity_matrix"
finput_description_sim_matrix = "Data/description_similarity_matrix"
finput_user_cluster_set = "Data/user_cluster_set"
finput_train_item_id = "Data/train_item_id"
finput_nonlinreg = "Data/nonlinreg"
finput_init_tp = 1.0
finput_init_dp = 1.0
foutput_iuclst_rating_matrix = "Data/iuclst_rating_matrix"
'''

if (__name__ == '__main__'):
	#### data path
	finput_iu_rating_matrix_train = sys.argv[1]
	finput_title_sim_matrix = sys.argv[2]
	finput_description_sim_matrix = sys.argv[3]
	finput_user_cluster_set = sys.argv[4]
	finput_train_item_id = sys.argv[5]
	finput_nonlinreg = sys.argv[6]
	finput_init_tp = float(sys.argv[7])
	finput_init_dp = float(sys.argv[8])
	foutput_iuclst_rating_matrix = sys.argv[9]

	# load data
	iu_rating_matrix_train = scipy.sparse.load_npz(finput_iu_rating_matrix_train)
	title_sim_matrix = rw.readffile(finput_title_sim_matrix)
	description_sim_matrix = rw.readffile(finput_description_sim_matrix)
	user_cluster_set = rw.readffile(finput_user_cluster_set)
	train_item_id = rw.readffile(finput_train_item_id)

	# run matlab script and get parameters for title and description
	print("call matlab script....")
	eng = matlab.engine.start_matlab()
	x = eng.my_fitnlm(finput_nonlinreg, finput_init_tp, finput_init_dp, nargout=3)
	theta1, theta2 = x[0], x[1]
	eng.quit()
	sim_matrix = theta1*title_sim_matrix + theta2*description_sim_matrix

	# extract similarity matrix for training item
	sim_matrix_train = sim_matrix.loc[train_item_id, train_item_id].values

	# user cluster - item rating matrix
	iuclst_rating_matrix = np.zeros((iu_rating_matrix_train.shape[0], len(user_cluster_set)))
	item_in_node = list(range(iu_rating_matrix_train.shape[0]))
	for ind, user_cluster in zip(range(len(user_cluster_set)), user_cluster_set):
		print("user cluster: %d / %d"%(ind+1, len(user_cluster_set)), end="\r")
		user_cluster_size = len(user_cluster)
		sub_rating_matrix = iu_rating_matrix_train[np.ix_(item_in_node, user_cluster)].T.toarray()     # user number * item number
		sub_rating_matrix_pred = (np.dot(sub_rating_matrix, sim_matrix_train) / np.dot(sub_rating_matrix != 0, sim_matrix_train)) # user number * item number
		iuclst_rating_matrix[:, ind] = np.sum(sub_rating_matrix + (sub_rating_matrix == 0) * sub_rating_matrix_pred, axis=0) / user_cluster_size
	print("user cluster/item rating matrix generated done!")

	rw.write2file(iuclst_rating_matrix, foutput_iuclst_rating_matrix)
	print("file saved done!")

