class DecisionTree:

	def __init__(self, iu_sparse_matrix, iuclst_rating_matrix, item_sim_matrix, user_cluster_set, depth):
		self.iu_sparse_matrix = iu_sparse_matrix
		self.iuclst_rating_matrix = iuclst_rating_matrix
		self.item_sim_matrix = item_sim_matrix
		self.user_cluster_set = user_cluster_set
		self.user_cluster_id_set = list(range(0, len(user_cluster_set)))
		self.item_num = iu_sparse_matrix.shape[0]
		self.user_num = iu_sparse_matrix.shape[1]
		self.tree = list(range(0, self.item_num))
		self.depth_threshold = depth
		self.node_interval = [[0, self.item_num-1]]



	def errorCalculation(self, item_in_node):
		sub_rating_matrix = self.iu_sparse_matrix[np.ix_(item_in_node)]
		mean = np.sum(sub_rating_matrix, axis=0) / (sub_rating_matrix.getnnz(axis=0) + 1e-5)
		deviation = np.array(sub_rating_matrix-mean) * (sub_rating_matrix !=0).toarray()
		return np.sum(deviation**2)


	def findOptUserCluster(self, cur_depth, cur_index):
		min_error = -1
		opt_item_in_left_child = []
		opt_item_in_middle_child = []
		opt_item_in_right_child = []
		item_in_node = self.tree[self.node_interval[cur_depth][cur_index][0]:self.node_interval[cur_depth][cur_index][1]]
		ratings = self.iuclst_rating_matrix[np.ix_(item_in_node)]
		for user_cluster_id in self.user_cluster_id_set:
			ratings_for_cluster = ratings[:, user_cluster_id]
			item_in_left_child = []
			item_in_middle_child = []
			item_in_right_child = []
			for i in range(ratings_for_cluster.shape[0]):
				if ratings_for_cluster[i] >= 4:
					item_in_right_child.append(item_in_node[i])
				elif ratings_for_cluster[i] <= 2.5:
					item_in_left_child.append(item_in_node[i])
				else:
					item_in_middle_child.append(item_in_node[i])

			error = self.errorCalculation(item_in_left_child) + self.errorCalculation(item_in_middle_child) + self.errorCalculation(item_in_right_child)
			if error < min_error:
				min_error = error
				opt_user_cluster_id = user_cluster_id
				opt_item_in_left_child = item_in_left_child[:]
				opt_item_in_middle_child = item_in_middle_child[:]
				opt_item_in_right_child = item_in_right_child[:]


		return [opt_item_in_left_child, opt_item_in_middle_child, opt_item_in_right_child], opt_user_cluster_id


	def dividToChild(self, optRes, cur_depth, cur_index):
		# update tree
		self.tree[self.node_interval[cur_depth][cur_index][0]:self.node_interval[cur_depth][cur_index][1]] = optRes[0] + optRes[1] + optRes[2]
		interval1 = self.node_interval[cur_depth][cur_index][0] + len(optRes[0]) - 1
		interval2 = interval1 + len(optRes[1])
		interval3 = interval2 + len(optRes[2])
		# left child interval
		self.node_interval[cur_depth+1][cur_index*3] = [self.node_interval[cur_depth][cur_index][0], interval1]
		# middle child interval
		self.node_interval[cur_depth+1][cur_index*3+1] = [interval1+1, interval2]
		# right child interval
		self.node_interval[cur_depth+1][cur_index*3+2] = [interval2+1, interval3]


	def treeConstruction(self, cur_depth, cur_index):

		# termination condition
		cur_depth += 1
		if cur_depth >= self.depth_threshold:
			return

		optRes, opt_user_cluster_id = self.findOptUserCluster()
		self.dividToChild(optRes, cur_depth, cur_index)
		self.user_cluster_id_set.remove(opt_user_cluster_id)
		# left child
		self.treeConstruction(cur_depth+1, cur_index*3)
		# middle child
		self.treeConstruction(cur_depth+1, cur_index*3+1)
		# right child
		self.treeConstruction(cur_depth+1, cur_index*3+2)
		self.user_cluster_id_set.append(opt_user_cluster_id)










