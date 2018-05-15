import numpy as np
import pandas as pd
from scipy.sparse import *
from MatrixFactorization import MatrixFactorization

class DecisionTree:

	def __init__(self, iu_sparse_matrix_train, iu_sparse_matrix_test, iuclst_rating_matrix_train, iuclst_rating_matrix_test, user_cluster_set, depth):
		self.iu_sparse_matrix_train = iu_sparse_matrix_train
		self.iu_sparse_matrix_test = iu_sparse_matrix_test
		self.iuclst_rating_matrix_train = iuclst_rating_matrix_train
		self.iuclst_rating_matrix_test = iuclst_rating_matrix_test
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

		# prediction model
		self.prediction_model = {}

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
		ratings = self.iuclst_rating_matrix_train[np.ix_(item_in_node)]

		if len(item_in_node) == 0:
			return [[], [], []], -1, -1, -1

		for user_cluster_id in self.user_cluster_id_set:
			# print(user_cluster_id)
			ratings_for_cluster = ratings[:, user_cluster_id]
			sorted_array = np.sort(ratings_for_cluster)
			node_size = len(sorted_array)
			itve1 = sorted_array[round(node_size/3)]
			itve2 = sorted_array[round((2*node_size)/3)]
			item_in_left_child = []
			item_in_middle_child = []
			item_in_right_child = []
			for i in range(ratings_for_cluster.shape[0]):
				if ratings_for_cluster[i] > itve2:
				# if ratings_for_cluster[i] >= 4:
					item_in_right_child.append(item_in_node[i])
				elif ratings_for_cluster[i] <= itve1:
				# elif ratings_for_cluster[i] <= 2.5:
					item_in_left_child.append(item_in_node[i])
				else:
					item_in_middle_child.append(item_in_node[i])

			error_dislike = self.errorCalculation(item_in_left_child)
			error_mediocre = self.errorCalculation(item_in_middle_child)
			error_like = self.errorCalculation(item_in_right_child)

			# if cur_depth == 0:
			# 	print("user_cluster_id:%d"%user_cluster_id)
			# 	print("error:%f"%(error_dislike+error_mediocre+error_like))
			# 	# print("error_dislike:%f"%error_dislike)
			# 	# print("error_like:%f"%error_like)
			# 	# print("error_mediocre:%f"%error_mediocre)
			# 	# print(list(ratings_for_cluster))
			# 	# print(ratings_for_cluster.shape[0])
			# 	print("\n")

			error = error_dislike + error_mediocre + error_like
			if min_error == -1 or error < min_error:
				min_error = error
				opt_user_cluster_id = user_cluster_id
				opt_itve1 = itve1
				opt_itve2 = itve2
				opt_item_in_left_child = item_in_left_child[:]
				opt_item_in_middle_child = item_in_middle_child[:]
				opt_item_in_right_child = item_in_right_child[:]
		# print("opt_user_cluster_id:%d"%(opt_user_cluster_id))
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



	def buildPredModel(self, params=[0.01], rank=10):
		min_rmse_dict = {}
		nonzero = self.iu_sparse_matrix_test.getnnz()
		nonzero_matrix = (self.iu_sparse_matrix_test != 0)
		MF = MatrixFactorization()
		for test_depth in range(self.depth_threshold):
			# if test_depth < 1:
			# 	continue
			print("level %d"%(test_depth))
			train_lst = []
			length = len(self.node_interval[test_depth])
			# generate input for spark ALS train
			for index, interval in zip(range(length), self.node_interval[test_depth]):
				print("%d/%d"%(index+1, length), end="\r")
				if interval[1] - interval[0] == -1:
					continue
				sub_rating_matrix = self.iu_sparse_matrix_train[np.ix_(self.tree[interval[0]:interval[1]+1])]
				# calculate average ratings for pseudo item to users
				avg_rating = np.sum(sub_rating_matrix, axis=0) / (1e-9+sub_rating_matrix.getnnz(axis=0))
				uid = avg_rating.nonzero()[1]
				rating = np.array(avg_rating[np.ix_([0], uid)])[0]
				for i in range(len(uid)):
					train_lst.append((uid[i], index, float(rating[i])))
			print("Rating Number of level " + str(test_depth) + ": " + str(len(train_lst)))
			# print(train_lst)

			# test different params for MF
			min_RMSE = -1
			self.prediction_model.setdefault(test_depth, {})
			for param in params:
				MF.change_parameter(regParam=param)
				#################################### Spark ####################################
				try:
					user_profile, item_profile = MF.matrix_factorization(train_lst)
				except:
					MF.end()
					MF = MatrixFactorization()
					MF.change_parameter(regParam=param)
					user_profile, item_profile = MF.matrix_factorization(train_lst)
				#################################### Spark ####################################

				################################ Calculate RMSE ##############################
				RMSE = self.predict(test_depth, item_profile, user_profile)
				print("Parameters: %f, RMSE: %f"%(param, RMSE))
				if min_RMSE == -1 or RMSE < min_RMSE:
					min_user_profile = user_profile
					min_item_profile = item_profile
					min_plambda = param
					min_RMSE = RMSE
				if RMSE > min_RMSE:
					break
				################################ Calculate RMSE ##############################

			# save the best profiles and param corresponding to each level
			print("min RMSE: %f"%min_RMSE)
			min_rmse_dict[test_depth] = min_RMSE
			self.prediction_model[test_depth]['upro'] = min_user_profile
			self.prediction_model[test_depth]['ipro'] = min_item_profile
			self.prediction_model[test_depth]['plambda'] = min_plambda
		MF.end()


	def predict(self, test_depth, item_profile, user_profile):
		self.prediction_model[test_depth]['ipro'] = item_profile
		P = np.dot(np.array(list(item_profile.values())), np.array(list(user_profile.values())).T)  
		# print(user_profile)
		P_test = np.zeros(self.iu_sparse_matrix_test.shape)
		rating_matrix_test_unqueried = self.iu_sparse_matrix_test.toarray()
		for itemid in range(self.iu_sparse_matrix_test.shape[0]):
			pred_index = 0
			final_level = 0
			rated_user = []
			user_all_ratings = self.iu_sparse_matrix_test[itemid, :].nonzero()[0]
			for depth in range(test_depth):
				rating = self.iuclst_rating_matrix_test[itemid][self.user_set_id[final_level][pred_index]]
				if rating > self.node_assc_rating[final_level][pred_index][1]:
					tmp_pred_index = 3*pred_index + 2
					if tmp_pred_index in self.prediction_model[depth+1]['ipro']:
						final_level += 1
						pred_index = tmp_pred_index
					else:
						break
				elif rating <= self.node_assc_rating[final_level][pred_index][0]:
					tmp_pred_index = 3*pred_index
					if tmp_pred_index in self.prediction_model[depth+1]['ipro']:
						rated_user.append(self.user_set_id[depth][pred_index])
						final_level += 1
						pred_index = tmp_pred_index
					else:
						break
				else:
					tmp_pred_index = 3*pred_index + 1
					if tmp_pred_index in self.prediction_model[depth+1]['ipro']:
						rated_user.append(self.user_set_id[depth][pred_index])
						final_level += 1
						pred_index = tmp_pred_index
					else:
						break      
			# print("pred_index before", pred_index)       
			pred_index = list(self.prediction_model[final_level]['ipro'].keys()).index(pred_index)
			# print("pred_index after", pred_index) 
			# print(P.shape)  
			# print("itemid:"+str(itemid)+" final_level:"+str(final_level)+" pred_index:"+str(pred_index))
			P_test[itemid, :] = P[pred_index, :]
			rating_matrix_test_unqueried[itemid, rated_user] = 0

		rating_matrix_test_unqueried = csc_matrix(rating_matrix_test_unqueried)
		P_test = (rating_matrix_test_unqueried!=0).multiply(P_test)
		P_test = P_test.tolil()
		P_test[P_test>5] = 5
		P_test[P_test<0] = 0
		diff = P_test - rating_matrix_test_unqueried
		return ( diff.multiply(diff).sum() / (rating_matrix_test_unqueried!=0).sum() )**0.5