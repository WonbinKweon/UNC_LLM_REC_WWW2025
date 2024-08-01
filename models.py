import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import torch.utils.data
import matplotlib.pyplot as plt
import time

class BPR(nn.Module):
    def __init__(self, num_user, num_item, emb_dim):
        super(BPR, self).__init__()
        self.num_user = num_user
        self.num_item = num_item

        self.user_emb = nn.Embedding(self.num_user, emb_dim)
        self.item_emb = nn.Embedding(self.num_item, emb_dim)

        nn.init.normal_(self.user_emb.weight, mean=0., std= 0.01)
        nn.init.normal_(self.item_emb.weight, mean=0., std= 0.01)
        
    # outputs logits
    def forward(self, batch_user, batch_pos_item, batch_neg_item):
        u = self.user_emb(batch_user)
        i = self.item_emb(batch_pos_item)
        j = self.item_emb(batch_neg_item)
        
        pos_score = (u * i).sum(dim=1, keepdim=True)
        neg_score = (u * j).sum(dim=1, keepdim=True)

        output = (pos_score, neg_score)
        
        return output

    def forward_pair(self, batch_user, batch_item):
        u = self.user_emb(batch_user)
        i = self.item_emb(batch_item)
        
        pos_score = (u * i).sum(dim=1, keepdim=True)
        
        return pos_score

    def forward_eval(self, batch_user):
        return torch.mm(self.user_emb(batch_user), self.item_emb.weight.data.T)

    def get_loss(self, output):
        return -(output[0] - output[1]).sigmoid().log().sum()
        #return -(output[0] - output[1]).sigmoid().log().mean()