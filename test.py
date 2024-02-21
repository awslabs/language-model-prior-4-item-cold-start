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

with open('user_item_all_cs.pkl', 'rb') as f:
    user_item = pickle.load(f)
with open('item_emb_cs.pkl', 'rb') as f:
    item_emb = pickle.load(f)

# with open('user_item_all_Prime_Pantry.pkl', 'rb') as f:
#     user_item = pickle.load(f)
# with open('item_emb_Prime_Pantry.pkl', 'rb') as f:
#     item_emb = pickle.load(f)
print(len(item_emb))
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
        self.item_emb_dim = 384
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
model.load_state_dict(torch.load("./model_save/SASRec_global.epoch=20.lr=0.001.layer=2.head=4.hidden=32.maxlen=100.rho=10.pth"))
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

for user in list(item_emb.keys()):
# for user in range(int(args.usernum*args.train_rate) + 1, args.usernum + 1):
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



# def evaluate_valid(model, dataset, args):
#     [train, valid, test, usernum, itemnum] = copy.deepcopy(dataset)

#     NDCG = 0.0
#     valid_user = 0.0
#     HT = 0.0
    
#     if usernum>10000:
#         users = random.sample(range(1, usernum + 1), 10000)
#     else:
#         users = range(1, usernum + 1)
        
#     for u in users:
#         if len(train[u]) < 1 or len(valid[u]) < 1: continue

#         seq = np.zeros([args.maxlen], dtype=np.int32)
#         idx = args.maxlen - 1
#         for i in reversed(train[u]):
#             seq[idx] = i
#             idx -= 1
#             if idx == -1: break

#         rated = set(train[u])
#         rated.add(0)
#         item_idx = [valid[u][0]]
#         for _ in range(100):
#             t = np.random.randint(1, itemnum + 1)
#             while t in rated: t = np.random.randint(1, itemnum + 1)
#             item_idx.append(t)

#         predictions = -model.predict(*[np.array(l) for l in [[u], [seq], item_idx]])
#         predictions = predictions[0]

#         rank = predictions.argsort().argsort()[0].item()

#         valid_user += 1

#         if rank < 10:
#             NDCG += 1 / np.log2(rank + 2)
#             HT += 1
#         if valid_user % 100 == 0:
#             print('.', end="")
#             sys.stdout.flush()

#     return NDCG / valid_user, HT / valid_user
