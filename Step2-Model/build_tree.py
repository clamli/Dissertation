import sys
import numpy as np
from DecisionTree import DecisionTree
import read_write as rw
import scipy.sparse

'''
finput_iu_rating_matrix_train = "Data/iu_sparse_matrix_train.npz"
finput_iuclst_rating_matrix = "Data/iuclst_rating_matrix"
finput_user_cluster_set = "Data/user_cluster_set"
finput_desired_depth = 5
'''

if (__name__ == '__main__'):

	finput_iu_sparse_matrix_train = sys.argv[1]
	finput_iu_sparse_matrix_test = sys.argv[2]
	finput_iuclst_rating_matrix_train = sys.argv[3]
	finput_iuclst_rating_matrix_test = sys.argv[4]
	finput_user_cluster_set = sys.argv[5]
	finput_desired_depth = int(sys.argv[6])

	# read into data for tree construction
	iu_sparse_matrix_train = scipy.sparse.load_npz(finput_iu_sparse_matrix_train)
	iu_sparse_matrix_test = scipy.sparse.load_npz(finput_iu_sparse_matrix_test)
	iuclst_rating_matrix_train = rw.readffile(finput_iuclst_rating_matrix_train)
	iuclst_rating_matrix_test = rw.readffile(finput_iuclst_rating_matrix_test)
	user_cluster_set = rw.readffile(finput_user_cluster_set)

	# build tree
	dt_model = DecisionTree(iu_sparse_matrix_train, iu_sparse_matrix_test, iuclst_rating_matrix_train, iuclst_rating_matrix_test, user_cluster_set, finput_desired_depth)
	dt_model.buildTreeModel()
	print("\ntree construction finished")
	# build prediction model
	dt_model.buildPredModel()
	print("prediction model finished")
	# predict
	# dt_model.predict()

	
	