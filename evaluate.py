# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

import os
import time
import torch
import argparse
import pickle
from model import SASRec
from utils import *

import os
import time
import torch
import argparse
import pickle
from model import SASRec
from utils import *
import torch.nn as nn
from numpy.linalg import inv
import matplotlib.pyplot as plt
import numpy as np


# with open('user_item_all.pkl', 'rb') as f:
#     user_item = pickle.load(f)
# with open('item_emb.pkl', 'rb') as f:
#     item_emb = pickle.load(f)
# with open('item_llm.pkl', 'rb') as f:
#     item_llm = pickle.load(f)
# with open('item_dist.pkl', 'rb') as f:
#     item_dist = pickle.load(f)
# with open('item_cov.pkl', 'rb') as f:
#     item_cov = pickle.load(f)
# item_llm = np.stack(item_llm)


def hit(gt_item, pred_items):
    if gt_item in pred_items:
        return 1
    return 0


def ndcg(gt_item, pred_items):
    if gt_item in pred_items:
        index = pred_items.index(gt_item)
        return np.reciprocal(np.log2(index+2))
    return 0


def metrics(model, test_loader, top_k):
    HR, NDCG = [], []

    for user, item_i, item_j in test_loader:
        user = user.cuda()
        item_i = item_i.cuda()
        item_j = item_j.cuda() # not useful when testing

        prediction_i, prediction_j, _ = model(user, item_i, item_j)
        _, indices = torch.topk(prediction_i, top_k)
        recommends = torch.take(
                item_i, indices).cpu().numpy().tolist()

        gt_item = item_i[0].item()
        HR.append(hit(gt_item, recommends))
        NDCG.append(ndcg(gt_item, recommends))

    return np.mean(HR), np.mean(NDCG)



with open('user_item_all_amazon.pkl', 'rb') as f:
    user_item = pickle.load(f)
with open('item_emb_amazon.pkl', 'rb') as f:
    item_emb = pickle.load(f)

print("Data loaded!")
class params:
    def __init__(self):
        self.num_epochs = 3
        self.usernum = len(user_item)
        self.maxlen = 100
        self.batch_size = 64
        self.train = False
        self.train_rate = 0.8
        self.itemnum = 209171 # not consequtive numbers, 62423 in total
        self.item_emb_dim = item_emb[1][0].shape[0]
        self.l2_emb = 0
        self.layer = 2
        self.num_heads = 4
        self.num_blocks = 2
        self.dropout_rate = 0.5
        self.hidden_units = 32
        self.lr = 0.001
        # self.rho = 0.0001
        self.rho = 0.001
        self.k = 10
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
args = params()

# sampler = WarpSampler(user_item, item_emb, args.usernum, args.itemnum, args.item_emb_dim, args.train, args.train_rate, batch_size=args.batch_size, maxlen=args.maxlen, n_workers=3)

print("Sampler made!")

model = SASRec(args.usernum, args.itemnum, args).to(args.device) # no ReLU activation in original SASRec implementation?
model.load_state_dict(torch.load("./model_save/SASRec_amazon.epoch=20.lr=0.001.layer=2.head=4.hidden=32.maxlen=100.rho=0.01.pth"))
model = model.float()
model.eval()

# _, seq, pos, _, seq_emb, pos_emb, neg_emb = sampler.next_batch() # tuples to ndarray
# seq, pos, seq_emb, pos_emb, neg_emb = np.array(seq), np.array(pos), np.array(seq_emb), np.array(pos_emb), np.array(neg_emb)

NDCG_10 = 0.0
HT_10 = 0.0
NDCG_20 = 0.0
HT_20 = 0.0
NDCG_40 = 0.0
HT_40 = 0.0
valid_user = 0.0
N = 40
cnt = 0

for user in range(int(args.usernum*args.train_rate) + 1, args.usernum + 1):
# for user in range(int(args.usernum*args.train_rate) + 1, int(args.usernum*args.train_rate) + 10):
    if cnt % 1000 == 0:
        print(cnt)
    
    cnt += 1
    seq = np.zeros([args.maxlen], dtype=np.int32)
    pos = np.zeros([args.maxlen], dtype=np.int32)
    nxt = user_item[user][-1]
    idx = args.maxlen - 1

    ts = set(user_item[user])
    for i in reversed(user_item[user][:-1]):
        seq[idx] = i
        pos[idx] = nxt
        nxt = i
        idx -= 1
        if idx == -1: break

    # Item embedding
    seq_emb = [np.zeros(args.item_emb_dim)] * args.maxlen
    pos_emb = [np.zeros(args.item_emb_dim)] * args.maxlen
    nxt = item_emb[user][-1]
    idx = args.maxlen - 1

    for i in reversed(item_emb[user][:-1]):
        seq_emb[idx] = i
        pos_emb[idx] = nxt
        nxt = i
        idx -= 1
        if idx == -1: break

    seq_emb = np.vstack(seq_emb)[np.newaxis, :, :]
    pos_emb = np.vstack(pos_emb)[np.newaxis, :, :]
    seq = seq[np.newaxis, :]
    
    # print(seq_emb.shape)
    # print(pos_emb.shape)
    
    predictions = -model.predict(seq, seq_emb, pos_emb)
    predictions = predictions[0]
    
    rank = predictions.argsort().argsort()[0].item()
    
    valid_user += 1
    
    if rank < 10:
        NDCG_10 += 1 / np.log2(rank + 2)
        HT_10 += 1
        
    if rank < 20:
        NDCG_20 += 1 / np.log2(rank + 2)
        HT_20 += 1
    
    if rank < 40:
        NDCG_40 += 1 / np.log2(rank + 2)
        HT_40 += 1

print("NDCG10: ", NDCG_10 / valid_user)
print("NDCG20: ", NDCG_20 / valid_user)
print("NDCG40: ", NDCG_40 / valid_user)
print("HT10: ", HT_10 / valid_user)
print("HT20: ", HT_20 / valid_user)
print("HT40: ", HT_40 / valid_user)
