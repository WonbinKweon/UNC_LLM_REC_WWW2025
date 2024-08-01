import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import torch.utils.data
import matplotlib.pyplot as plt
import scipy
import math

def swor_exp(w, R):
    n = len(w)
    E = -np.log(np.random.uniform(0,1,size=(R,n))) 
    E /= w
    return np.argsort(E, axis=1)

def p_perm(w, z):
    "The probability of a permutation `z` under the sampling without replacement scheme."
    # assert len(z) <= len(w)
    p = 1.0
    W = w.sum()
    for t in range(len(z)):
        x = w[z[t]]
        p *= x / W
        W -= x
    return p

def compute_rank_entropy(y_prob_matrix, mc, n):
    y_prob_matrix += 1e-10
    
    rank_entropy = []
    for y_prob in y_prob_matrix:
        permutations = swor_exp(y_prob, mc)[:, :n]
        entropy = 0
        for per in permutations:
            entropy -= np.log2(p_perm(y_prob, per))
        rank_entropy.append(entropy/mc)
        
    return np.asarray(rank_entropy)

def compute_entropy(vec: np.ndarray):
    vec = vec + 1e-10
    shape = list(vec.shape)
    shape[-1] = -1
    vec = vec / np.sum(vec, axis=-1).reshape(shape)
    entropy = -np.sum(vec * np.log2(vec), axis=-1)
    return entropy

############### For Model training ###############
def is_visited(base_dict, user_id, item_id):
    if user_id in base_dict and item_id in base_dict[user_id]:
        return True
    else:
        return False

def get_user_item_count_dict(interactions):
    user_count_dict = {}
    item_count_dict = {}

    for user, item in interactions:
        if user not in user_count_dict:
            user_count_dict[user] = 1
        else:
            user_count_dict[user] += 1

        if item not in item_count_dict:
            item_count_dict[item] = 1
        else:
            item_count_dict[item] += 1

    return user_count_dict, item_count_dict

class traindset(torch.utils.data.Dataset):
    def __init__(self, num_user, num_item, train_dic, train_pair, num_neg, user_list, item_list):
        super(traindset, self).__init__()
        self.num_user = num_user
        self.num_item = num_item
        self.num_neg = num_neg
        self.train_dic = train_dic
        self.train_pair = train_pair
        
        self.user_list = user_list
        self.item_list = item_list
                         
    def negative_sampling(self, safe):      
        sample_list = np.random.choice(list(range(max(self.item_list)+1)), size = safe * len(self.train_pair) * self.num_neg)

        self.train_arr = []
        bookmark = 0
        for user in self.user_list:
            train_list = self.train_dic[user]
            num_train = len(train_list) * self.num_neg

            neg_list = sample_list[bookmark:bookmark+num_train]
            bookmark = bookmark+num_train
            _, mask, _ = np.intersect1d(neg_list, train_list, return_indices=True)

            while True:
                if len(mask) == 0:
                    break
                neg_list[mask] = sample_list[bookmark:bookmark+len(mask)]
                bookmark = bookmark+len(mask)
                _, mask, _ = np.intersect1d(neg_list, train_list, return_indices=True)

            for i,j in zip(train_list, neg_list):
                self.train_arr.append((user, i, j))
       
        self.train_arr = np.array(self.train_arr)

    def __len__(self):
        return len(self.train_pair) * self.num_neg

    def __getitem__(self, idx):
        return self.train_arr[idx][0], self.train_arr[idx][1], self.train_arr[idx][2]

    
def evaluate(KS, topk_matrix, test_dic, num_item, user_list, reduce=True, only_NDCG=False):
    num_user = topk_matrix.shape[0]

    idcg_ = [sum([1/np.log2(l+2) for l in range(K)]) for K in range(1, max(KS)+1)]
    idcg_matrix = np.tile(idcg_, num_user).reshape(num_user, max(KS))
    for idx, u in enumerate(user_list):
        idcg_matrix[idx][len(test_dic[u])-1:] = idcg_matrix[idx][min(max(KS)-1, len(test_dic[u])-1)]

    dcg_ = np.array([1/np.log2(K+2) for K in range(max(KS))])  
    dcg_matrix = topk_matrix * dcg_
    dcg_sum_matrix = np.cumsum(dcg_matrix, axis=-1)    
    ndcg_matrix = dcg_sum_matrix / idcg_matrix
    NDCG = ndcg_matrix[:, np.array(KS)-1]
    
    np.set_printoptions(precision=4)
    if only_NDCG:
        if reduce:
            return np.mean(NDCG, axis=0, keepdims=True)
        else:
            return NDCG
    
    hr_sum_matrix = np.cumsum(topk_matrix, axis=-1)
    HR = []
    F1 = []
    for idx, u in enumerate(user_list):
        item_pos = test_dic[u] # pos item idx

        HR.append([hr_sum_matrix[idx][K-1] / min(len(item_pos), K) for K in KS])

        Pre_u = np.asarray([hr_sum_matrix[idx][K-1] / K for K in KS])
        Rec_u = np.asarray([hr_sum_matrix[idx][K-1] / len(item_pos) for K in KS])
        F1.append((2*Pre_u*Rec_u / (Pre_u + Rec_u + 0.000001)).tolist())
            
    if reduce:
        return np.mean(NDCG, axis=0, keepdims=True), np.mean(np.asarray(HR), axis=0, keepdims=True), np.mean(np.asarray(F1), axis=0, keepdims=True)
    else:
        return NDCG, np.asarray(HR), np.asarray(F1) 