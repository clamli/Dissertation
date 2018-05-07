from scipy.sparse import *
import numpy as np

class DecisionTreeModel:
    def __init__(self, source, depth_threshold=10, plambda=7, MSP_item=200):
        
        self.sMatrix = source
        self.depth_threshold = depth_threshold
        self.plambda = plambda
        self.MSP_item = MSP_item
        self.real_item_num = self.sMatrix.shape[0]
        self.global_mean = self.sMatrix.sum()/self.sMatrix.getnnz()
        x = find(source)
        itemset = x[0]
        userset = x[1]

        #### Calculate rate of progress ####
        self.cur_depth = 0
        self.node_num = 0
        self.cur_node = 0
        for i in range(self.depth_threshold):
            self.node_num += 3 ** i
        
        #### Initiate Tree, lr_bound ####
        self.tree = list(range(self.sMatrix.shape[1]))
        self.split_item = []
        self.lr_bound = {'0': [[0, len(self.tree) - 1]]}

        #### Generate rU ####        
        self.rU = {}        
        num_ratings = len(userset)
        i = 0
        for itemid, userid in zip(itemset, userset):
            # put approximate 5000 user in each file. Divide user num with 5000.
            if i%100000 == 0:
                print("%.2f%%" %(100 * i/num_ratings))
            i += 1
            self.rU.setdefault(userid, {})[itemid] = int(source[itemid, userid])        
        print("rU Generation DONE")
         
        #### Generate bias, sum_cur_t, sum_2_cur_t, sum_cntt ####
        self.biasU = np.zeros(self.sMatrix.shape[1])
        self.sum_cur_t = np.zeros(self.real_item_num)
        self.sum_2_cur_t = np.zeros(self.real_item_num)
        self.sum_cntt = np.zeros(self.real_item_num)
        i = 0
        for userid in self.tree:
            if i % 50000 == 0:
                print("%.2f%%" % (100 * i / (0.75 * 480189)))
            i += 1

            self.biasU[userid] = (self.sMatrix.getcol(userid).sum() \
                                     + self.plambda * self.global_mean) /   \
                                 (self.plambda + self.sMatrix.getcol(userid).getnnz())
            user_all_rating_id = self.sMatrix.getcol(userid).nonzero()[0]
            user_all_rating = find(self.sMatrix.getcol(userid))[2]
            self.sum_cur_t[user_all_rating_id[:]] += user_all_rating[:] - self.biasU[userid]
            self.sum_2_cur_t[user_all_rating_id[:]] += (user_all_rating[:] - self.biasU[userid]) ** 2
            self.sum_cntt[user_all_rating_id[:]] += 1  
        print("bias, sum_cur_t, sum_2_cur_t Generation DONE")
        
        # initialize
        self.item_size = self.sMatrix.shape[0]
        self.user_size = len(self.tree)        
        self.MPS = []
        print("Initiation DONE!")


    def calculate_error(self, sumt, sumt_2, cntt):
        ''' Calculate error for one item-split in one node '''
        Error_i = np.sum(sumt_2 - (sumt ** 2) / (cntt + 1e-9))
        return Error_i

    def generate_decision_tree(self, lr_bound_for_node, chosen_id):
        
        #### Show Rating Progress ####
        for i in range(self.cur_depth - 1):
            print("┃", end="")
        print("┏", end="")
        self.cur_node += 1
        print("Current depth: " + str(self.cur_depth) + "        %.2f%%" % (100 * self.cur_node / self.node_num))
        
        #### Terminate ####
        self.cur_depth += 1
        if self.cur_depth >= self.depth_threshold or len(chosen_id) == self.item_size:
            return   
        
        #### Choose Most Popular Items of This Node ####     
        num_rec = np.zeros(self.item_size)
        for userid in self.tree[lr_bound_for_node[0]:(lr_bound_for_node[1] + 1)]:
            user_all_rating_id = np.array(list(self.rU[userid].keys()))
            num_rec[user_all_rating_id] += 1
        MPS_item_id = list(np.argsort(-num_rec)[:])
        for item_id in chosen_id:
            MPS_item_id.remove(item_id)
        MPS_item_id = MPS_item_id[:self.MSP_item]
            
        #### Find optimum item to split ####
        min_sumtL, min_sumtD, min_sumtL_2, min_sumtD_2, min_sumtU, min_sumtU_2, Error = {}, {}, {}, {}, {}, {}, {}
        min_Error = "None"
        for itemid in MPS_item_id:
            if itemid in chosen_id:
                continue
            '''
                user_rating_item_in_nodet: np.array([ [uid01, rating01], [uid02, rating02], ... ])
                to find all users in node t who rates item i
            '''
            
            user_rating_item_in_nodet = np.array([[userid, self.rU[userid][itemid]] for userid in
                                         self.tree[lr_bound_for_node[0]:(lr_bound_for_node[1] + 1)] if
                                         itemid in self.rU[userid]])         
            sumt = np.zeros((self.item_size, 3))
            sumt_2 = np.zeros((self.item_size, 3))
            cntt = np.zeros((self.item_size, 3))
            for user in user_rating_item_in_nodet:
                ''' user_all_rating: array [ [itemid11, rating11], [itemid12, rating12], ... ] '''
                user_all_rating_id = np.array(list(self.rU[user[0]].keys()))
                user_all_rating = np.array(list(self.rU[user[0]].values()))
                #### calculate sumtL for node LIKE ####
                if user[1] > 3:
#                     split.setdefault(itemid, []).append(user[0])
                    sumt[user_all_rating_id[:], 0] += user_all_rating[:] - self.biasU[user[0]]
                    sumt_2[user_all_rating_id[:], 0] += (user_all_rating[:] - self.biasU[user[0]]) ** 2
                    cntt[user_all_rating_id[:], 0] += 1
                #### calculate sumtD for node DISLIKE ####
                elif user[1] <= 3:
                    sumt[user_all_rating_id[:], 1] += user_all_rating[:] - self.biasU[user[0]]
                    sumt_2[user_all_rating_id[:], 1] += (user_all_rating[:] - self.biasU[user[0]]) ** 2
                    cntt[user_all_rating_id[:], 1] += 1

            #### calculate sumtU for node UNKNOWN ####
            sumt[:, 2] = self.sum_cur_t[:] - sumt[:, 0] - sumt[:, 1]
            sumt_2[:, 2] = self.sum_2_cur_t[:] - sumt_2[:, 0] - sumt_2[:, 1]
            cntt[:, 2] = self.sum_cntt[:] - cntt[:, 0] - cntt[:, 1]
            Error[itemid] = self.calculate_error(sumt, sumt_2, cntt)
            if min_Error == "None" or Error[itemid] < min_Error:
                min_sumt = sumt
                min_sumt_2 = sumt_2
                min_cntt = cntt
                min_Error = Error[itemid]
        #### Find optimum split-item ####
        optimum_itemid = min(Error, key=Error.get)
        if len(self.split_item) == self.cur_depth - 1:
            self.split_item.append([optimum_itemid])
        else:
            self.split_item[self.cur_depth - 1].append(optimum_itemid)
        chosen_id.append(optimum_itemid)
#         print(Error)
        # print("split item found!")
#         print(optimum_itemid)
        #### sort tree ####
        self.lr_bound.setdefault(str(self.cur_depth), []).append([])  # for LIKE
        self.lr_bound[str(self.cur_depth)].append([])  # for DISLIKE
        self.lr_bound[str(self.cur_depth)].append([])  # for UNKNOWN
        listU, listL, listD = [], [], []
        for userid in self.tree[lr_bound_for_node[0]:(lr_bound_for_node[1] + 1)]:
            if optimum_itemid not in self.rU[userid]:
                listU.append(userid)
            elif self.rU[userid][optimum_itemid] > 3:
                listL.append(userid)
            elif self.rU[userid][optimum_itemid] <= 3:
                listD.append(userid)
        self.tree[lr_bound_for_node[0]:(lr_bound_for_node[1] + 1)] = listL + listD + listU
        self.lr_bound[str(self.cur_depth)][-3] = [lr_bound_for_node[0],
                                                  lr_bound_for_node[0] + len(listL) - 1]  # for LIKE
        self.lr_bound[str(self.cur_depth)][-2] = [lr_bound_for_node[0] + len(listL),
                                                  lr_bound_for_node[0] + len(listL) + len(listD) - 1]  # for DISLIKE
        self.lr_bound[str(self.cur_depth)][-1] = [lr_bound_for_node[0] + len(listL) + len(listD),
                                                  lr_bound_for_node[0] + len(listL) + len(listD) + len(listU) - 1]  # for UNKNOWN

        #### Generate Subtree of Node LIKE ####
        self.sum_cur_t = min_sumt[:, 0]
        self.sum_2_cur_t = min_sumt_2[:, 0]
        self.sum_cntt = min_cntt[:, 0]
        self.generate_decision_tree(self.lr_bound[str(self.cur_depth)][-3], chosen_id[:])
        self.cur_depth -= 1

        #### Generate Subtree of Node DISLIKE ####
        self.sum_cur_t = min_sumt[:, 1]
        self.sum_2_cur_t = min_sumt_2[:, 1]
        self.sum_cntt = min_cntt[:, 1]
        self.generate_decision_tree(self.lr_bound[str(self.cur_depth)][-2], chosen_id[:])
        self.cur_depth -= 1

        #### Generate Subtree of Node UNKNOWN ####
        self.sum_cur_t = min_sumt[:, 2]
        self.sum_2_cur_t = min_sumt_2[:, 2]
        self.sum_cntt = min_cntt[:, 2]
        self.generate_decision_tree(self.lr_bound[str(self.cur_depth)][-1], chosen_id[:])
        self.cur_depth -= 1

    def build_model(self):
        #### Construct the tree & get the prediction model ####
        self.generate_decision_tree(self.lr_bound['0'][0], [])