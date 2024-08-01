import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import pandas as pd
from torch.autograd import Variable
import random
import torch.optim as optim
import pickle
import torch.utils.data
from torch.backends import cudnn
from scipy.sparse import csr_matrix
import math
import bottleneck as bn
import time
import matplotlib.pyplot as plt
import argparse

from models import BPR
from utils import *
from models import *

def train(args):
    #print(vars(args))
    gpu = args.cuda
    
    ## dataset ##
    dataset = args.dataset
    train_pair = np.load('data/'+dataset+'/train.npy', allow_pickle=True) # N*2
    train_dic = np.load('data/'+dataset+'/train_dic.npy', allow_pickle=True).item() # dict of (user - item list)
    val_dic = np.load('data/'+dataset+'/val_dic.npy', allow_pickle=True).item()
    trainval_dic = np.load('data/'+dataset+'/trainval_dic.npy', allow_pickle=True).item()
    test_dic = np.load('data/'+dataset+'/test_dic.npy', allow_pickle=True).item()

    idx_title = np.load('data/'+dataset+'/idx_title.npy', allow_pickle=True).item()
    # idx_title_year = np.load('data/'+dataset+'/idx_title_year.npy', allow_pickle=True).item()
    # idx_genre = np.load('data/'+dataset+'/idx_genre.npy', allow_pickle=True).item()

    user_list = list(train_dic.keys())
    item_list = list(idx_title.keys())

    num_user = max(user_list)+1 #len(user_list)
    num_item = max(item_list)+1 #len(item_list)

    ## hyper-parameters ##
    epochss = [100]
    verbose = 10 # evaluate

    emb_dims = [128]
    batch_sizes = [8192]
    num_neg = args.nneg
    lrs = [1e-3]
    wds = [1e-5]

    for epochs in epochss:
        for emb_dim in emb_dims:
            for batch_size in batch_sizes:
                for lr in lrs:
                    for wd in wds:
                        print('**', epochs, emb_dim, batch_size, lr, wd)
                        train_dataset = traindset(num_user, num_item, train_dic, train_pair, num_neg, user_list, item_list)
                        train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
                        val_loader = torch.utils.data.DataLoader(user_list, batch_size=args.bs_eval)
                        
                        model = BPR(max(user_list)+1, max(item_list)+1, emb_dim)
                        model = model.cuda(gpu)
                        optimizer = optim.Adam(model.parameters(), lr = lr, weight_decay=wd)
                        
                        ### training
                        for epoch in range(0,epochs+1):
                            model.train()
                            t0 = time.time() ##
                            
                            train_loader.dataset.negative_sampling(safe=args.safe)
                            t_neg = time.time() ##
                            
                            loss_train = np.zeros(1)
                            
                            for ub, ib, jb in train_loader:
                                output = model(ub.cuda(gpu), ib.cuda(gpu), jb.cuda(gpu))
                                loss_CF = model.get_loss(output)

                                optimizer.zero_grad()
                                loss_CF.backward()
                                optimizer.step()

                                loss_train[0] += loss_CF.cpu().tolist() 
                            loss_train /= len(train_loader)
                            
                            t_train = time.time() ##
                            
                            ### validation
                            if epoch % verbose == 0:
                                print('epoch = {}, loss = {:.3f}, time = {:.4f}, {:.4f}'.format(epoch, loss_train[0], t_neg-t0, t_train-t_neg))
                                ## val, test
                                t0 = time.time() ##
                                with torch.no_grad():                
                                    model.eval()
                                    N = 100
                                    topk_matrix_val = np.zeros((num_user, N))
                                    topk_matrix_test = np.zeros((num_user, N))

                                    for us in val_loader:
                                        ## inference
                                        row = model.forward_eval(us.cuda(gpu))
                                        
                                        ## masking
                                        row_val = row.clone().detach()
                                        row_test = row.clone().detach()
                                        for idx, u in enumerate(us): # do not recommend interacted items
                                            row_val[idx][train_dic[u.numpy().item()]] = float('-inf')
                                            row_test[idx][trainval_dic[u.numpy().item()]] = float('-inf')

                                        ## ranking (sorting)
                                        _, topk_idx_val = torch.topk(row_val, N)
                                        _, topk_idx_test = torch.topk(row_test, N)

                                        ## boolean matrix (|U| * N)
                                        interactions_val = torch.zeros([us.size()[0], num_item], dtype=torch.bool, device=gpu)
                                        interactions_test = torch.zeros([us.size()[0], num_item], dtype=torch.bool, device=gpu)
                                        users_t, items_t, users_v, items_v = [], [], [], []
                                        for idx, u in enumerate(us):
                                            u = u.cpu().numpy().item()
                                            for i in val_dic[u]:
                                                users_v.append(idx)
                                                items_v.append(i)
                                            for i in test_dic[u]:
                                                users_t.append(idx)
                                                items_t.append(i)                                                
                                                
                                        interactions_val[users_v, items_v] = True  
                                        interactions_test[users_t, items_t] = True        
                                        y_sorted_val = interactions_val.gather(-1, topk_idx_val)
                                        y_sorted_test = interactions_test.gather(-1, topk_idx_test)

                                        topk_matrix_val[us] = y_sorted_val.cpu().numpy()
                                        topk_matrix_test[us] = y_sorted_test.cpu().numpy()

                                    t_rec = time.time() ##
                                    NDCG, HR, F1 = evaluate([1,5,10,20,50,70,100], topk_matrix_val, val_dic, num_item, user_list)
                                    t_eval = time.time() ##
                                    print('NDCG={}, HR={}, time ={:.2f}, {:.2f}'.format(NDCG, HR, t_rec-t0, t_eval-t_rec))
                                    
                                    NDCG, HR, F1 = evaluate([1,5,10,20,50,70,100], topk_matrix_test, test_dic, num_item, user_list)
                                    t_eval2 = time.time() ##
                                    print('NDCG={}, HR={}, time ={:.2f}'.format(NDCG, HR, t_eval2-t_eval))

                                    del topk_matrix_val
                                    del topk_matrix_test
                                        
    if args.save:
        if args.save_name == None:
            torch.save(model, 'model/'+args.model+'_'+args.dataset)
        else:
            torch.save(model, 'model/'+args.save_name)        
                    
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--cuda', default=0, type=int) ## gpu number
    parser.add_argument('--dataset', default='ml-1m', type=str)
    parser.add_argument('--safe', default=2, type=int) ## for negative sampling
    parser.add_argument('--model', default='BPR', type=str) ## BPR
    parser.add_argument('--nlayer', default=2, type=int) ## num layer
    parser.add_argument('--save', default=1, type=int) ## save
    parser.add_argument('--save_name', default=None, type=str) ## save file name
    parser.add_argument('--nneg', default=1, type=int) ## num neg
    parser.add_argument('--pen', default=0, type=int) ## penalized DCG & NDCG
    parser.add_argument('--bs_eval', default=50, type=int) ## batch size for evaluation
    
    
    args = parser.parse_args()

    result = train(args)