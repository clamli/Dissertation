import numpy as np
import pandas as pd
from MatrixFactorization import MatrixFactorization

class DecisionTree:

	def __init__(self, iu_sparse_matrix_train, iu_sparse_matrix_test, iuclst_rating_matrix, user_cluster_set, depth):
		self.iu_sparse_matrix_train = iu_sparse_matrix_train
		self.iu_sparse_matrix_test = iu_sparse_matrix_test
		self.iuclst_rating_matrix = iuclst_rating_matrix
		self.user_cluster_set = user_cluster_set
		self.user_cluster_id_set = list(range(0, len(user_cluster_set)))
		self.item_num = iu_sparse_matrix_train.shape[0]
		self.user_num = iu_sparse_matrix_train.shape[1]
		self.tree = list(range(0, self.item_num))
		self.depth_threshold = depth
		self.node_interval = [[] for i in range(depth)]
		self.node_interval[0].append([0, self.item_num-1])
		self.user_set_id = [[] for i in range(depth-1)]
		self.node_assc_rating = [[] for i in range(depth-1)]
		# progress record
		self.cur_node = 1
		self.node_num = 0
		for i in range(depth):
			self.node_num += 3**i

		# pseudo item and user profile
		self.pseudo_item = {}
		self.user_profile = {}

	def errorCalculation(self, item_in_node):
		sub_rating_matrix = self.iu_sparse_matrix_train[np.ix_(item_in_node)]
		sum_node = np.sum(sub_rating_matrix, axis=0)
		sum_node_2 = np.sum(sub_rating_matrix.power(2), axis=0)
		num_node = np.sum(sub_rating_matrix != 0, axis=0)
		deviation = np.sum(sum_node_2 - np.power(sum_node, 2)/(num_node+1e-9))
		return deviation


	def findOptUserCluster(self, cur_depth, cur_index):
		min_error = -1
		opt_item_in_left_child = []
		opt_item_in_middle_child = []
		opt_item_in_right_child = []
		item_in_node = self.tree[self.node_interval[cur_depth][cur_index][0]:self.node_interval[cur_depth][cur_index][1]+1]
		ratings = self.iuclst_rating_matrix[np.ix_(item_in_node)]

		if len(item_in_node) == 0:
			return [[], [], []], -1, -1, -1

		for user_cluster_id in self.user_cluster_id_set:
			# print(user_cluster_id)
			ratings_for_cluster = ratings[:, user_cluster_id]
			sorted_array = np.sort(ratings_for_cluster)
			itve1 = sorted_array[int(len(sorted_array)/3)]
			itve2 = sorted_array[int((2*len(sorted_array))/3)]
			item_in_left_child = []
			item_in_middle_child = []
			item_in_right_child = []
			for i in range(ratings_for_cluster.shape[0]):
				# if ratings_for_cluster[i] >= itve2:
				if ratings_for_cluster[i] >= 4:
					item_in_right_child.append(item_in_node[i])
				# elif ratings_for_cluster[i] <= itve1:
				elif ratings_for_cluster[i] <= 2.5:
					item_in_left_child.append(item_in_node[i])
				else:
					item_in_middle_child.append(item_in_node[i])

			error_dislike = self.errorCalculation(item_in_left_child)
			error_mediocre = self.errorCalculation(item_in_middle_child)
			error_like = self.errorCalculation(item_in_right_child)

			error = error_dislike + error_mediocre + error_like
			if min_error == -1 or error < min_error:
				min_error = error
				opt_user_cluster_id = user_cluster_id
				opt_itve1 = itve1
				opt_itve2 = itve2
				opt_item_in_left_child = item_in_left_child[:]
				opt_item_in_middle_child = item_in_middle_child[:]
				opt_item_in_right_child = item_in_right_child[:]

		return [opt_item_in_left_child, opt_item_in_middle_child, opt_item_in_right_child], opt_user_cluster_id, opt_itve1, opt_itve2


	def dividToChild(self, optRes, cur_depth, cur_index):
		# update tree
		self.tree[self.node_interval[cur_depth][cur_index][0]:self.node_interval[cur_depth][cur_index][1]+1] = optRes[0] + optRes[1] + optRes[2]
		if len(self.node_interval[cur_depth+1]) == 0:
			begin = 0
		else:
			begin = self.node_interval[cur_depth+1][-1][1] + 1
		interval1 = begin + len(optRes[0]) - 1
		interval2 = interval1 + len(optRes[1])
		interval3 = interval2 + len(optRes[2])
		# left child interval
		self.node_interval[cur_depth+1].append([begin, interval1])
		# middle child interval
		self.node_interval[cur_depth+1].append([interval1+1, interval2])
		# right child interval
		self.node_interval[cur_depth+1].append([interval2+1, interval3])


	def treeConstruction(self, cur_depth, cur_index):

		# progress record
		print('Current depth: %d        %.2f%%'%(cur_depth+1, 100*self.cur_node/self.node_num), end="\r")
		# termination condition
		if cur_depth >= self.depth_threshold - 1:
			return
		self.cur_node += 3

		# opt_itve1 -> left node; opt_itve2 -> right node
		optRes, opt_user_cluster_id, opt_itve1, opt_itve2 = self.findOptUserCluster(cur_depth, cur_index)
		self.user_set_id[cur_depth].append(opt_user_cluster_id)
		self.node_assc_rating[cur_depth].append([opt_itve1, opt_itve2])
		self.dividToChild(optRes, cur_depth, cur_index)
		
		if opt_user_cluster_id != -1:
			self.user_cluster_id_set.remove(opt_user_cluster_id)
		# left child
		self.treeConstruction(cur_depth+1, cur_index*3)
		# middle child
		self.treeConstruction(cur_depth+1, cur_index*3+1)
		# right child
		self.treeConstruction(cur_depth+1, cur_index*3+2)
		self.user_cluster_id_set.append(opt_user_cluster_id)


	def buildTreeModel(self):
		self.treeConstruction(0, 0)



	def buildPredModel(self):
		for test_depth in range(self.depth_threshold):
			print("level %d"%(test_depth+1))
			self.pseudo_item[test_depth] = pd.DataFrame(columns=['item_id', 'pseudo_item_profile'])
			self.user_profile[test_depth] = pd.DataFrame(columns=['user_profile'])
			train_lst = []
			length = len(self.node_interval[test_depth])

			# generate input for spark ALS train
			for index, interval in zip(range(length), self.node_interval[test_depth]):
				print("%d/%d"%(index+1, length), end="\r")
				cur_depth = test_depth
				cur_index = index
				while interval[1] - interval[0] == -1:
					cur_depth -= 1
					cur_index = int(cur_index/3)
					interval = self.node_interval[cur_depth][cur_index]
				self.pseudo_item[test_depth].loc[index] = [self.tree[interval[0]:interval[1]+1], []]

				sub_rating_matrix = self.iu_sparse_matrix_train[np.ix_(self.tree[interval[0]:interval[1]+1])]
				# calculate average ratings for pseudo item to users
				avg_rating = np.sum(sub_rating_matrix, axis=0) / (1e-5+sub_rating_matrix.getnnz(axis=0))
				uid = avg_rating.nonzero()[1]
				rating = np.array(avg_rating[np.ix_([0], uid)])[0]
				for i in range(len(uid)):
					train_lst.append((uid[i], index, float(rating[i])))

			#################################### Spark ####################################
			MF = MatrixFactorization()
			try:
				user_feature, item_feature = MF.matrix_factorization(train_lst)
			except:
				MF.end()
				MF = MatrixFactorization()
				user_feature, item_feature = MF.matrix_factorization(train_lst)
			length = len(item_feature)
			for i, each in zip(range(length), item_feature):
				print("item profiles: %d/%d"%(i+1, length), end="\r")
				self.pseudo_item[test_depth].loc[i][1] = np.array(each[1])
			print("\n")
			length = len(user_feature)
			for i, each in zip(range(length), user_feature):
				print("user profiles: %d/%d"%(i+1, length), end="\r")
				self.user_profile[test_depth].loc[i] = [each[1].tolist()]
			print("\n")
			MF.end()
			#################################### Spark ####################################


	def predict(self):
		iuclst_rating_matrix_test = self.iuclst_rating_matrix[self.item_num:, :]
		iu_pred_ratings_test = np.zeros(self.iu_sparse_matrix_test.shape)
		length = iuclst_rating_matrix_test.shape[0]
		for test_depth in range(self.depth_threshold):
			for i in range(iuclst_rating_matrix_test.shape[0]):
				print("prediction: %d/%d"%(i+1, length), end="\r")
				cur_depth = 0
				cur_index = 0
				# print(self.node_interval)
				while self.node_interval[cur_depth][cur_index][1] - self.node_interval[cur_depth][cur_index][1] != -1:
					pre_depth = cur_depth
					pre_index = cur_index
					if cur_depth == test_depth:
						break
					rating = iuclst_rating_matrix_test[i][self.user_set_id[cur_depth][cur_index]]
					# if rating >= self.node_assc_rating[cur_depth][cur_index][1]:   # right
					if rating >= 4:
						cur_index = cur_index*3 + 2
					# elif rating <= self.node_assc_rating[cur_depth][cur_index][0]:   # left
					elif rating <= 2.5:
						cur_index = cur_index*3
					else:     					# middle
						cur_index = cur_index*3 + 1
					cur_depth += 1
				iu_pred_ratings_test[i, :] = np.dot(np.array(list(self.user_profile[test_depth]['user_profile'])), self.pseudo_item[test_depth].iloc[pre_index]['pseudo_item_profile'])
			
			# calculate RMSE
			iu_true_ratings_test = self.iu_sparse_matrix_test.toarray()
			RMSE = (np.sum(((iu_true_ratings_test != 0) * iu_pred_ratings_test - iu_true_ratings_test)**2) / np.sum(iu_true_ratings_test != 0))**0.5
			print("level %d: %f"%(test_depth+1, RMSE))