# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0


import os
import time
import torch
import argparse
import pickle
from model import SASRec
from utils import *
import pandas as pd
from sentence_transformers import SentenceTransformer
from scipy.spatial import distance
from numpy.linalg import inv

model = SentenceTransformer('all-MiniLM-L6-v2')

# with open('user_item_all_Prime_Pantry.pkl', 'rb') as f:
#     user_item = pickle.load(f)
# with open('item_emb_Prime_Pantry.pkl', 'rb') as f:
#     item_emb = pickle.load(f)
# with open('item_llm_Prime_Pantry.pkl', 'rb') as f:
#     item_llm = pickle.load(f)

with open('user_item_all.pkl', 'rb') as f:
    user_item = pickle.load(f)
with open('item_emb.pkl', 'rb') as f:
    item_emb = pickle.load(f)

print("Data loaded!")
# movies = pd.read_csv("./amazon/meta_Prime_Pantry.csv")
movies = pd.read_csv("./data/movies.csv")
# items = movies.title.unique().tolist()
items = movies.genres.unique().tolist()
print(items[:5])
print(len(items))
item_llm = []
i = 0
for content in items:
    i += 1
    if i % 1000 == 0:
        print(i)
    try:
        item_llm.append(model.encode(content))
    except:
        pass
    

print(len(item_llm))

with open("item_llm.pkl", "wb") as fp:   #Pickling
    pickle.dump(item_llm, fp)

visited = set()
items = []
item_unique = {}
for i in range(1, len(user_item) + 1):
    if i % 1000 == 0:
        print(i)
    for j in range(len(user_item[i])):
        if user_item[i][j] in visited:
            continue
        
        visited.add(user_item[i][j])
        items.append(user_item[i][j])
        item_unique[user_item[i][j]] = item_emb[i][j]

print(len(item_unique))
print(len(items))

item_dist = {}
item_cov = {}
i = 0
for item in items:
    if i % 100 == 0:
        print(i)
    i += 1
    
    idx = [i for i in range(len(item_llm))]
    dist = [distance.euclidean(item_unique[item], y) for y in item_llm]
    
    # rank
    idx_rank = [x for _, x in sorted(zip(dist, idx))]
    item_dist[item] = idx_rank[:71] # sqrt(#items)
    
    # cov
    # item_emb_tmp = [item_llm[i] for i in item_dist[item]]
    # item_emb_tmp.append(np.array(item_unique[item]))
    # mu = np.mean(np.vstack(item_emb_tmp), axis = 0)
    # cov = (1 / 250) * sum([np.matmul((a - mu)[:, np.newaxis], (a - mu)[np.newaxis, :]) for a in item_emb_tmp])
    # item_cov[item] = inv(cov +np.eye(384)*1e-3)


with open('item_dist.pkl', 'wb') as f:
    pickle.dump(item_dist, f)
    
# with open('item_cov.pkl', 'wb') as f:
#     pickle.dump(item_cov, f)
