# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0


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

with open('user_item_all.pkl', 'rb') as f:
    user_item = pickle.load(f)
with open('item_emb.pkl', 'rb') as f:
    item_emb = pickle.load(f)
with open('item_llm.pkl', 'rb') as f:
    item_llm = pickle.load(f)
with open('item_dist.pkl', 'rb') as f:
    item_dist = pickle.load(f)
# with open('item_cov.pkl', 'rb') as f:
#     item_cov = pickle.load(f)
item_llm = np.stack(item_llm)

print("Data loaded!")
class params:
    def __init__(self):
        self.num_epochs = 20
        self.usernum = len(user_item)
        self.maxlen = 100
        self.batch_size = 64
        self.train = True
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
        self.rho =0.01
        self.k = 71
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
args = params()

num_batch = len(user_item) // args.batch_size # tail? + ((len(user_train) % args.batch_size) != 0)

sampler = WarpSampler(user_item, item_emb, args.usernum, args.itemnum, args.item_emb_dim, args.train, args.train_rate, batch_size=args.batch_size, maxlen=args.maxlen, n_workers=3)

print("Sampler made!")

model = SASRec(args.usernum, args.itemnum, args).to(args.device) # no ReLU activation in original SASRec implementation?
model = model.float()
for name, param in model.named_parameters():
    try:
        torch.nn.init.xavier_normal_(param.data)
    except:
        pass # just ignore those failed init layers

model.train() # enable model training

epoch_start_idx = 1

bce_criterion = torch.nn.BCEWithLogitsLoss() # torch.nn.BCELoss()
adam_optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, betas=(0.9, 0.98))

T = 0.0
t0 = time.time()

loss_best = 10000000000
loss_rec = 0

loss_rcd = []
loss_reg_rcd = []
loss_pos_rcd = []
loss_neg_rcd = []

for epoch in range(epoch_start_idx, args.num_epochs + 1):

    for step in range(num_batch): # tqdm(range(num_batch), total=num_batch, ncols=70, leave=False, unit='b'):
        _, seq, pos, seq_emb, pos_emb, neg_emb = sampler.next_batch() # tuples to ndarray
        seq, pos, seq_emb, pos_emb, neg_emb = np.array(seq), np.array(pos), np.array(seq_emb), np.array(pos_emb), np.array(neg_emb)

        pos_logits, neg_logits, seqs_out = model(seq, seq_emb, pos_emb, neg_emb)
        pos_labels, neg_labels = torch.ones(pos_logits.shape, device=args.device), torch.zeros(neg_logits.shape, device=args.device)

        # regularization
        # z
        llm_emb_z = model.item_emb(torch.from_numpy(item_llm).to(args.device))
        ## item sequence
        seq = seq.flatten()
        idx = np.where(seq != 0)
        seq = seq[idx].tolist()
        ## Trained embedding
        seqs_out = seqs_out.reshape(-1, args.hidden_units)
        seqs_out = seqs_out[idx, :].squeeze() # len(seq), hidden_units
        ## Trained embedding of nearest items
        seq_z = torch.stack([llm_emb_z[item_dist[item][:args.k], :] for item in seq]).reshape(len(seq), args.k, -1) # len(seq), k, hidden_units
        seq_z = torch.sum((seqs_out.unsqueeze(1) - seq_z)**2, dim = -1) # len(seq), k
        seq_z = seq_z[:, 1:]
        
        # x
        ## LLM embedding
        seq_emb = seq_emb.reshape(-1, 384)
        seq_emb = seq_emb[idx, :]
        ## Cov
        cov = np.stack([np.std(item_llm[item_dist[item][:args.k], :], axis = 0) for item in seq]) # len(seq), 384
        cov = 1 / (cov + 1e-3)
        cov = np.repeat(cov[:, np.newaxis, :], args.k, axis = 1) # len(seq), k, 384
        ## LLM embedding of nearest items
        seq_x = np.stack([item_llm[item_dist[item][:args.k], :] for item in seq]).reshape(len(seq), args.k, -1) # len(seq), k, 384
        seq_x = np.sum(((seq_emb.squeeze()[:, np.newaxis, :] - seq_x)**2) * cov, axis = -1) # len(seq), k
        seq_x = torch.exp(-0.5*torch.from_numpy(seq_x)).to(args.device)
        seq_x = seq_x[:, 1:]

        loss_reg = torch.sum(seq_x * seq_z)

        adam_optimizer.zero_grad()
        indices = np.where(pos != 0)
        loss_pos = bce_criterion(pos_logits[indices], pos_labels[indices])
        loss_neg = bce_criterion(neg_logits[indices], neg_labels[indices])
        loss = loss_pos + loss_neg + args.rho * loss_reg
        
        loss_rcd.append(loss.detach().cpu().numpy().item())
        loss_reg_rcd.append(loss_reg.detach().cpu().numpy().item())
        loss_pos_rcd.append(loss_pos.detach().cpu().numpy().item())
        loss_neg_rcd.append(loss_neg.detach().cpu().numpy().item())
        
        for param in model.item_emb.parameters(): 
            loss += args.l2_emb * torch.norm(param)
            
        loss_rec += loss.item()
        loss.backward()
        adam_optimizer.step()
        print("loss in epoch {} iteration {}: {} loss_reg={} loss_pos={} loss_neg={}".format(epoch, step, loss.item(), loss_reg.item(), loss_pos.item(), loss_neg.item())) # expected 0.4~0.6 after init few epochs
        
    loss_rec /= num_batch
    
    if loss_rec < loss_best:
        folder = "./model_save"
        fname = 'SASRec.epoch={}.lr={}.layer={}.head={}.hidden={}.maxlen={}.rho={}.pth'.format(args.num_epochs, args.lr, args.num_blocks, args.num_heads, args.hidden_units, args.maxlen, args.rho)
        # fname = fname.format(args.num_epochs, args.lr, args.num_blocks, args.num_heads, args.hidden_units, args.maxlen)
        torch.save(model.state_dict(), os.path.join(folder, fname))
    loss_rec = 0
    

plt.figure()
plt.plot(loss_rcd)
plt.savefig('./model_save/loss.epoch={}.lr={}.layer={}.head={}.hidden={}.maxlen={}.rho={}.png'.format(args.num_epochs, args.lr, args.num_blocks, args.num_heads, args.hidden_units, args.maxlen, args.rho))

plt.figure()
plt.plot(loss_pos_rcd)
plt.savefig('./model_save/loss_pos.epoch={}.lr={}.layer={}.head={}.hidden={}.maxlen={}.rho={}.png'.format(args.num_epochs, args.lr, args.num_blocks, args.num_heads, args.hidden_units, args.maxlen, args.rho))

plt.figure()
plt.plot(loss_neg_rcd)
plt.savefig('./model_save/loss_neg.epoch={}.lr={}.layer={}.head={}.hidden={}.maxlen={}.rho={}.png'.format(args.num_epochs, args.lr, args.num_blocks, args.num_heads, args.hidden_units, args.maxlen, args.rho))

plt.figure()
plt.plot(loss_reg_rcd)
plt.savefig('./model_save/loss_reg.epoch={}.lr={}.layer={}.head={}.hidden={}.maxlen={}.rho={}.png'.format(args.num_epochs, args.lr, args.num_blocks, args.num_heads, args.hidden_units, args.maxlen, args.rho))
