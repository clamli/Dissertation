import sys
import pandas as pd
import numpy as np
import scipy.sparse
from scipy.sparse import csr_matrix, find
import scipy.io as sio
import gzip
import string
import read_write as rw

'''
finput_title = "Data/title_similarity_matrix"
finput_description = "Data/description_similarity_matrix"
finput_train_id = "Data/train_item_id"
finput_test_id = "Data/test_item_id"
finput_rating_matrix_train = "Data/iu_sparse_matrix_train.npz"
finput_rating_matrix_test = "Data/iu_sparse_matrix_test.npz"
foutput = "Data/nonlinreg.mat"
'''
if (__name__ == '__main__'):

	#### data path
	finput_title = sys.argv[1]
	finput_description = sys.argv[2]
	finput_train_id = sys.argv[3]
	finput_test_id = sys.argv[4]
	finput_rating_matrix_train = sys.argv[5]
	finput_rating_matrix_test = sys.argv[6]
	foutput = sys.argv[7]

	# read into similarity file and train/test item id
	matrix_title = rw.readffile(finput_title)
	matrix_description = rw.readffile(finput_description)
	train_id = rw.readffile(finput_train_id)
	test_id = rw.readffile(finput_test_id)

	# combine these items and select corresponding matrix
	item_id = list(train_id.keys()) + list(test_id.keys())
	matrix_title = matrix_title.loc[item_id, item_id]
	matrix_description = matrix_description.loc[item_id, item_id]

	# read into train/test rating sparse matrix and combine them up
	rating_matrix_train = scipy.sparse.load_npz(finput_rating_matrix_train)
	rating_matrix_test = scipy.sparse.load_npz(finput_rating_matrix_test)
	rating_matrix = scipy.sparse.csr_matrix(np.vstack((rating_matrix_train.toarray(),rating_matrix_test.toarray())))

	# generate argument pairs for non linear regression
	x = find(rating_matrix)
	st_r = []
	sd_r = []
	st = []
	sd = []
	ratings = []
	item_num = rating_matrix.shape[0]
	length = x[0].shape[0]
	cnt = 0
	for iid, uid, rating in zip(x[0], x[1], x[2]):
		cnt += 1
		print("progress: %d / %d"%(cnt, length), end="\r")
		flag = np.ones(item_num).reshape(1, item_num)                                # 1 * item number
		flag[0, iid] = 0
		ur_history = rating_matrix[:, uid].T.toarray() * flag                        # 1 * item number
		if ur_history.any() == False:
		    continue

		it_similarity = np.array(matrix_title.iloc[:, iid]).reshape(1, item_num)           # 1 * item number
		id_similarity = np.array(matrix_description.iloc[:, iid]).reshape(1, item_num)     # 1 * item number
		st_r.append(np.dot(ur_history, it_similarity.T)[0,0])
		sd_r.append(np.dot(ur_history, id_similarity.T)[0,0])
		st.append((it_similarity*(ur_history!=0)).sum())
		sd.append((id_similarity*(ur_history!=0)).sum())
		ratings.append(rating)
	sio.savemat(foutput, {'st_r': st_r,'sd_r': sd_r,'st': st, 'sd': sd, 'ratings' : ratings})
	print("\nfile saved done!")
