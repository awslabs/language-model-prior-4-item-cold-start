# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

import pandas as pd
from sentence_transformers import SentenceTransformer
import pickle

model = SentenceTransformer('all-MiniLM-L6-v2')

rate = pd.read_csv("./data/ratings.csv") 
movie = pd.read_csv("./data/movies.csv") 
print("Data loaded")

users = rate.userId.unique()
user_item = {}
items = {}

i = 0
for user in users:
    if i % 1000 == 0:
        print(i)
    i += 1
    user_item[user] = rate.loc[rate["userId"] == user].movieId.tolist()
    items[user] = []
    for item in user_item[user]:
        meta = movie.loc[movie["movieId"] == item].iloc[0]["genres"]
        item_emb = model.encode(meta)
        items[user].append(item_emb)
    if i % 1000 == 0:
        with open('user_item_all.pkl', 'wb') as f:
            pickle.dump(user_item, f)

        with open('item_emb.pkl', 'wb') as f:
            pickle.dump(items, f)

# print(user_item)
# print(items)
with open('user_item_all.pkl', 'wb') as f:
    pickle.dump(user_item, f)
    
with open('item_emb.pkl', 'wb') as f:
    pickle.dump(items, f)
