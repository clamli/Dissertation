import sys
import pandas as pd
import numpy as np
import gzip
import scipy.sparse
from scipy.sparse import csr_matrix, find
import random
import read2df as rdf
import read_write as rw

'''
	finput = "../Dataset/All_Beauty/reviews_All_Beauty.json.gz"
	finput_item = "Data/title"
	foutput1 = "Data/iu_sparse_matrix_train.npz"
	foutput2 = "Data/iu_sparse_matrix_test.npz"
	foutput_uid = "Data/uid"
	foutput_train_item_id = "Data/train_item_id"
	foutput_test_item_id = "Data/test_item_id"
'''
if (__name__ == '__main__'):

	#### data path
	finput = sys.argv[1]
	finput_item = sys.argv[2]
	foutput1 = sys.argv[3]
	foutput2 = sys.argv[4]
	foutput_uid = sys.argv[5]
	foutput_train_item_id = sys.argv[6]
	foutput_test_item_id = sys.argv[7]

	# read into item id whose title and description is not null
	dict_item_id = rw.readffile(finput_item)

	# read into review file and select item
	df = rdf.getDF(finput)
	df = df.loc[df['asin'].isin(dict_item_id)]

	# split item into train and test
	itemid = list(df['asin'].unique())
	train_item_id = random.sample(itemid, int(0.75*len(itemid)))
	test_item_id = [ele for ele in itemid if ele not in train_item_id]
	print("train: %d/%d, test: %d/%d"%(len(train_item_id), len(itemid), len(test_item_id), len(itemid)))


	df_train = df.loc[df['asin'].isin(train_item_id)]
	df_test = df.loc[df['asin'].isin(test_item_id)]
	# set user set as those who rate at least one item in the training set
	userid = list(set(list(df_train['reviewerID'])))
	print("user number: ", len(userid))

	# map user/item to id
	user_id_dict = {}
	for i in range(len(userid)):
		user_id_dict[userid[i]] = i
	train_item_id_dict = {}
	for i in range(len(train_item_id)):
		train_item_id_dict[train_item_id[i]] = i
	test_item_id_dict = {}
	for i in range(len(test_item_id)):
		test_item_id_dict[test_item_id[i]] = i
	col = len(userid)
	train_row = len(train_item_id)
	test_row = len(test_item_id)

	# transfer ratings to array
	# train
	iu_matrix_train = np.zeros((train_row, col), dtype=np.int8)
	cnt = 0
	lenght = df_train.shape[0]
	for index, row in df_train.iterrows():
		print("iu train matrix: %d / %d"%(cnt, lenght), end="\r")
		iu_matrix_train[train_item_id_dict[row['asin']], user_id_dict[row['reviewerID']]] = int(row['overall'])
		cnt += 1
	iu_sparse_matrix_train = scipy.sparse.csr_matrix(iu_matrix_train)
	print("density of iu train matrix is: %.4f%%"%(100*len(find(iu_sparse_matrix_train)[0])/(iu_sparse_matrix_train.shape[0]*iu_sparse_matrix_train.shape[1])))
	scipy.sparse.save_npz(foutput1, iu_sparse_matrix_train)
	# test
	iu_matrix_test = np.zeros((test_row, col), dtype=np.int8)
	cnt = 0
	lenght = df_test.shape[0]
	for index, row in df_test.iterrows():
		print("iu test matrix: %d / %d"%(cnt, lenght), end="\r")
		if row['reviewerID'] in user_id_dict.keys():
			iu_matrix_test[test_item_id_dict[row['asin']], user_id_dict[row['reviewerID']]] = int(row['overall'])
		cnt += 1
	iu_sparse_matrix_test = scipy.sparse.csr_matrix(iu_matrix_test)
	print("density of iu test matrix is: %.4f%%"%(100*len(find(iu_sparse_matrix_test)[0])/(iu_sparse_matrix_test.shape[0]*iu_sparse_matrix_test.shape[1])))
	scipy.sparse.save_npz(foutput2, iu_sparse_matrix_test)
	print("iu matrix generated done!")


	# write uid, train_item_id and test_item_id into files
	rw.write2file(user_id_dict, foutput_uid)
	rw.write2file(train_item_id_dict, foutput_train_item_id)
	rw.write2file(test_item_id_dict, foutput_test_item_id)
	print("write done!")