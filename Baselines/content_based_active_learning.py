import sys
import read_write as rw
import numpy as np
import scipy.sparse
from MatrixFactorization import MatrixFactorization

if (__name__ == '__main__'):
    finput_dataset = sys.argv[1]
    finput_K = (int)(sys.argv[2])
    iu_matrix_train_path = "../Data/" + finput_dataset + "/iu_sparse_matrix_train.npz"
    iu_matrix_test_path = "../Data/" + finput_dataset + "/iu_sparse_matrix_test.npz"
    train_item_id_path = "../Data/" + finput_dataset + "/train_item_id"
    test_item_id_path = "../Data/" + finput_dataset + "/test_item_id"
    item_sim_matrix_path = "../Data/" + finput_dataset + "/description_similarity_matrix"     # pass

    ui_matrix_train = scipy.sparse.load_npz(iu_matrix_train_path).T
    ui_matrix_test = scipy.sparse.load_npz(iu_matrix_test_path).T
    ui_matrix = scipy.sparse.csr_matrix(np.hstack((ui_matrix_train.toarray(), np.zeros(ui_matrix_test.shape))))
    train_item_id = rw.readffile(train_item_id_path)
    test_item_id = rw.readffile(test_item_id_path)
    item_sim_matrix = rw.readffile(item_sim_matrix_path)

    # Computing Score for user (Score = [user number, new item number])
    Score = (ui_matrix_train * item_sim_matrix.loc[train_item_id, test_item_id]) / \
            ((ui_matrix_train != 0) * item_sim_matrix.loc[train_item_id, test_item_id])

    # Active Learning
    train_item_num = len(train_item_id)
    ui_matrix = ui_matrix.tolil()
    ui_matrix_test = ui_matrix_test.tolil()
    for i in range(len(test_item_id)):
        ind = np.argsort(-Score[:, i])
        if finput_K < ind.shape[0]:
            topK = ind[:(finput_K+1)]
        else:
            topK = ind
        ui_matrix[topK, i+train_item_num] = ui_matrix_test[topK, i]
        ui_matrix_test[topK, i] = 0

    # Matrix Factorization
    nonzero = scipy.sparse.find(ui_matrix)
    train_lst = []
    for uid, itemid, rating in zip(nonzero[0], nonzero[1], nonzero[2]):
        train_lst.append((uid, itemid, float(rating)))
    MF = MatrixFactorization(usernum=ui_matrix.shape[0], itemnum=ui_matrix.shape[1])
    try:
        user_profile, item_profile = MF.matrix_factorization(train_lst)
    except:
        MF.end()
        MF = MatrixFactorization()
        user_profile, item_profile = MF.matrix_factorization(train_lst)
    pred_rating = np.dot(user_profile, item_profile[train_item_num:, :].T)
    nonzero_num = ui_matrix_test.getnnz()
    ui_matrix_test_arr = ui_matrix_test.toarray()
    RMSE = np.sum(((ui_matrix_test_arr != 0)*(pred_rating - ui_matrix_test_arr))**2 / nonzero_num)**0.5
    print("RMSE: %.4f"%RMSE)
    MF.end()

    
    

