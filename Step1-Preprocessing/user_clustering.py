import sys
import pandas as pd
import numpy as np
import read_write as rw
import k_medoids as km

'''
finput_user_similarity = "Data/user_similarity_matrix"
finput_cluster_number = 200
foutput_user_cluster_set = "Data/user_cluster_set"
'''

if (__name__ == '__main__'):
	# data path
	finput_user_similarity = sys.argv[1]
	finput_cluster_number = int(sys.argv[2])
	foutput_user_cluster_set = sys.argv[3]

	# read into user similarity matrix
	user_similarity_matrix = rw.readffile(finput_user_similarity)

	# k-medoids
	user_cluster_set = km.k_medoids(user_similarity_matrix.values, K=finput_cluster_number, max_iterations=20)
	print("\ndone!")

	rw.write2file(user_cluster_set, foutput_user_cluster_set)
	print("file saved done!")

	print("top 20% of user cluster:")
	length = []
	for lst in user_cluster_set:
		length.append(len(lst))
	length.sort(reverse=True)
	print(length[0:int(len(length)*0.2)])