#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import utils.dataset as ds
import sys
import utils.bm25 as bm25
from tqdm import tqdm
import numpy as np
import torch
from collections import defaultdict
import re

metadata = ds.get_metadata()
gold = ds.get_gold()
queries = ds.get_queries()
precedients = ds.get_precedients()
sections = ds.get_sections()


# In[ ]:


import argparse

parser = argparse.ArgumentParser(description="BM25")
parser.add_argument("--query_type", type=str, help="Query Type")
parser.add_argument("--candidate_type", type=str, help="Candidate Type")
parser.add_argument("--n_gram", type=int, help="N Gram")
#parser.add_argument("--output_path", type=int, help="Output Path")
args = parser.parse_args()


# In[ ]:


query_type = args.query_type
candidate_type = args.candidate_type
n_gram = args.n_gram
# query_type = "test"
# can_type = "secs"
# n_gram = 2
if candidate_type == "secs":
    candidates = sections
elif candidate_type == "prec":
    candidates = precedients
else:
    raise ValueError("candidate type should be secs or prec")


# In[ ]:


query_text ={}
for id in metadata[query_type]:
    count = 0
    for para in list(queries[id]["text"]):
        query_text[id+"/"+str(count)] = para[1]
        count += 1  

candidate_text ={}
for id in list(candidates[id]["text"]):
    count = 0
    for para in metadata[id]:
        candidate_text[id+"/"+str(count)] = para[1]
        count += 1 


# In[ ]:


qry2idx = {q: i for i,q in enumerate(metadata[query_type])}
idx2qry = {i: q for i,q in enumerate(metadata[query_type])}
can2idx = {p: i for i,p in enumerate(metadata[candidate_type])}
idx2can = {i: p for i,p in enumerate(metadata[candidate_type])}


# In[ ]:


def flatten(paras):
    text = defaultdict(list)
    for p, pt in paras.items():
        text[p.split('/')[0]].append(pt)
    text = {k: re.sub(' +', ' ', ' '.join(v).replace('\n', ' ').replace('\t', ' ')) for k,v in text.items()}
    return text

qry_txt_flat = flatten(query_text)
prec_txt_flat = flatten(candidate_text)
len(qry_txt_flat), len(prec_txt_flat)


# In[ ]:


qry_txt_list = [qry_txt_flat[idx2qry[i]] for i in range(len(idx2qry))]
prec_txt_list = [prec_txt_flat[idx2can[i]] for i in range(len(idx2can))]

bm25_precs = bm25.BM25(n_gram=n_gram)
bm25_precs.fit(prec_txt_list)

prec_scores = []
for qt in tqdm(qry_txt_list, desc=f"Calculating BM25 {n_gram}-gram {query_type} to {candidate_type} scores"):
    prec_scores.append(bm25_precs.transform(qt))

prec_scores = np.stack(prec_scores, axis=0)

torch.save(prec_scores, f'scores/bm25_{query_type}_{candidate_type}_{n_gram}gram.pt')
#torch.save(prec_scores, args.ouput_path)

