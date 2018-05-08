import time
import sys
import klepto
import shelve
import pickle
import numpy as np
from scipy.sparse import *
import matplotlib.pyplot as plt
import read_write as rw
from FactorizedDecisionTree import DecisionTreeModel
from MatrixFactorization import MatrixFactorization


if (__name__ == '__main__'):
	# data path
	finput_dataset = sys.argv[1]
	finput_depth = (int)(sys.argv[2])

	ui_matrix_train_csc = load_npz('../../Data/'+ finput_dataset + '/iu_sparse_matrix_train.npz').tocsc().T
	ui_matrix_test_csc = load_npz('../../Data/'+ finput_dataset + '/iu_sparse_matrix_test.npz').tocsc().T
	print("file load DONE")

	# build tree
	dtmodel = DecisionTreeModel(ui_matrix_train_csc, finput_depth)
	dtmodel.build_model()

	# parameter training
	split_item = dtmodel.split_item
	lr_bound = dtmodel.lr_bound
	tree = dtmodel.tree
	depth_threshold = dtmodel.depth_threshold
	lambda_list = [0.005, 0.025, 0.05, 0.075, 0.10, 0.11, 0.12, 0.13, 0.14, 0.15]
	MF = MatrixFactorization()
	min_rmse_list = []
	rmst_dict = {}
	prediction_model = {}
	ui_matrix_train = ui_matrix_train_csc.toarray()
	nonzero_matrix = (ui_matrix_train != 0)
	nonzero = ui_matrix_train_csc.getnnz()
	for level in range(depth_threshold):
		level = str(level)
		print("level:", level)
		prediction_model.setdefault(level, {})
		prediction_model[level]['item_id'] = []
		train_lst = []       
		for pseudo_item_bound, itemid in zip(lr_bound[level], range(len(lr_bound[level]))):
			if pseudo_item_bound[0] > pseudo_item_bound[1]:
				continue
			pseudo_item_lst = tree[pseudo_item_bound[0]:(pseudo_item_bound[1] + 1)]
			prediction_model[level]['item_id'].append(pseudo_item_lst)
			pseudo_matrix = np.array(ui_matrix_train_csc[:, pseudo_item_lst].sum(axis=1))[:,0] / \
						(ui_matrix_train_csc[:, pseudo_item_lst].getnnz(axis=1)+1e-9)
			train_lst += [(itemid, userid, float(pseudo_matrix[userid])) \
						for userid in range(pseudo_matrix.shape[0]) if pseudo_matrix[userid]]        

		print("Rating Number of level " + level + ": " + str(len(train_lst)))

		#### Train MF and Do validation ####
		min_RMSE = -1
		for plambda in lambda_list:
			#### Designate desired lambda here if required ####

			#### ----------------------------------------- ####
			print("Current plambda: " + str(plambda))
			MF.change_parameter(plambda)
			user_profile, item_profile = MF.matrix_factorization(train_lst)
			#         prediction_model[level]['upro'], prediction_model[level]['ipro'], prediction_model[level]['plambda'] \
			#                                          = user_profile, item_profile, plambda
			ipro = list(item_profile.keys())
			user_prof = np.array(list(user_profile.values()))
			pseudo_item_prof = np.array(list(item_profile.values()))
			P = np.dot(user_prof, pseudo_item_prof.T)
			item_prof = np.zeros((ui_matrix_train_csc.shape[1], pseudo_item_prof.shape[1]))
			for itemid, ind in zip(prediction_model[level]['item_id'], range(pseudo_item_prof.shape[0])):
				item_prof[itemid, :] = pseudo_item_prof[ind, :]
			RMSE = (np.sum((nonzero_matrix * np.dot(user_prof, item_prof.T) - ui_matrix_train)**2) / nonzero)**0.5
			print("Current RMSE: " + str(RMSE))
			rmst_dict.setdefault(level, []).append(RMSE)
			if min_RMSE == -1 or RMSE < min_RMSE:
				min_RMSE = RMSE
				min_item_profile = ipro
				min_P = P
				min_lambda = plambda
				
		min_rmse_list.append(min_RMSE)
		prediction_model[level]['ipro'] = min_item_profile
		prediction_model[level]['P'] = min_P
		prediction_model[level]['plambda'] = min_lambda
		print("min RMSE: " + str(min(rmst_dict[level])))
		# plt.figure(1)
		# plt.title('RMSE for level ' + level)
		# plt.xlabel('plambda')
		# plt.ylabel('RMSE')
		# plt.plot(lambda_list, rmst_dict[level])
		# plt.show()
	MF.end()

	# plot result
	plt.figure(1)
	plt.xlabel('level')
	plt.ylabel('min RMSE')
	plt.plot(list(range(len(min_rmse_list))), min_rmse_list)
	plt.show()

	# dump result
	rw.write2file(min_rmse_list, "./train_rmse_list")


	# test
	nominator = 0
	denominator = 0
	level = min_rmse_list.index(min(min_rmse_list))
	P_test = np.zeros(ui_matrix_test_csc.shape)
	rating_matrix_test_unqueried = ui_matrix_test_csc.toarray()
	for itemid in range(ui_matrix_test_csc.shape[1]):
	#         if userid % 100 == 0:
	#             print("%.2f%%" % (100 * userid / rating_matrix_csc_test.shape[1]))  
		pred_index = 0
		final_level = 0
		user_all_ratings = ui_matrix_test_csc[:,itemid].nonzero()[0]

		for depth in range(int(level)):
			if split_item[depth][pred_index] not in user_all_ratings:
				tmp_pred_index = 3*pred_index + 2
				if tmp_pred_index in prediction_model[str(depth+1)]['ipro']:
					final_level += 1
					pred_index = tmp_pred_index
				else:
					break
			elif ui_matrix_test_csc[split_item[depth][pred_index], itemid] > 3:
				tmp_pred_index = 3*pred_index
				if tmp_pred_index in prediction_model[str(depth+1)]['ipro']:
					final_level += 1
					pred_index = tmp_pred_index
				else:
					break
			elif ui_matrix_test_csc[split_item[depth][pred_index], itemid] <= 3:
				tmp_pred_index = 3*pred_index + 1
				if tmp_pred_index in prediction_model[str(depth+1)]['ipro']:
					final_level += 1
					pred_index = tmp_pred_index
				else:
					break
			
		pred_index = prediction_model[str(final_level)]['ipro'].index(pred_index)
		P_test[:, itemid] = prediction_model[str(final_level)]['P'][:, pred_index]

	# calculate test RMSE
	rating_matrix_test_unqueried = csc_matrix(rating_matrix_test_unqueried)
	P_test = (rating_matrix_test_unqueried!=0).multiply(P_test)
	nominator = (P_test - rating_matrix_test_unqueried).power(2).sum()
	denominator = (rating_matrix_test_unqueried!=0).sum()
	test_RMSE = (nominator/denominator)**0.5
	print('Test RMSE: %f'%test_RMSE)