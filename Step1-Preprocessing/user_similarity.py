import sys
import numpy as np
import pandas as pd
import scipy.sparse
from scipy.sparse import csr_matrix, find
import read_write as rw

'''
finput_uid = "Data/uid"
finput_rating_matrix_train = "Data/iu_sparse_matrix_train.npz"
foutput_user_similarity = "Data/user_similarity_matrix"
'''

if (__name__ == '__main__'):
	#### data path
	finput_uid = sys.argv[1]
	finput_rating_matrix_train = sys.argv[2]
	foutput_user_similarity = sys.argv[3]

	# read into user id information and train rating matrix
	uid = rw.readffile(finput_uid)
	rating_matrix_train = scipy.sparse.load_npz(finput_rating_matrix_train).toarray()

	# generate user similarity
	rating_matrix_train = (rating_matrix_train - np.sum(rating_matrix_train, axis=0) / np.sum(rating_matrix_train != 0, axis=0)) * (rating_matrix_train!=0)
	rating_matrix_train_2 = rating_matrix_train**2
	# user_similarity_matrix = np.dot(rating_matrix_train.T, rating_matrix_train) / (np.dot(rating_matrix_train_2.T, rating_matrix_train_2)**0.5 + 1e-9)
	row_num = rating_matrix_train.shape[0]
	col_num = rating_matrix_train.shape[1]
	user_similarity_matrix = np.zeros((col_num, col_num))
	nominatorM = np.dot(rating_matrix_train.T, rating_matrix_train)
	print("nominator done!")
	cnt = 0
	for i in range(col_num):
		cnt += 1
		print("progress: %d / %d"%(cnt, col_num), end="\r")
		flag = ((rating_matrix_train[:, i]!=0).reshape(row_num, 1))*(rating_matrix_train!=0)
		user_similarity_matrix[i] = nominatorM[i] / ((np.dot(rating_matrix_train_2[:, i].T, flag)**0.5) * (np.sum(rating_matrix_train_2*flag, axis=0)**0.5) + 1e-9)
	# or it will be 0 for some users
	# np.fill_diagonal(user_similarity_matrix, 1)
	print("\ndone!")

	# transfer to dataframe and save to file
	# rw.write2file(user_similarity_matrix, "Data/test")
	df_user_similarity_matrix = pd.DataFrame(user_similarity_matrix,index=list(uid.keys()),columns=list(uid.keys()))
	del user_similarity_matrix
	rw.write2file(df_user_similarity_matrix, foutput_user_similarity)
	print("file saved done!")