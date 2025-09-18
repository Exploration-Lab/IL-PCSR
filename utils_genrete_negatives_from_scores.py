#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import argparse

parser = argparse.ArgumentParser(description="Generate Negatives")
parser.add_argument("--query_type", type=str, help="Query Type")
parser.add_argument("--scores_path", type=str, help="Query Type")
parser.add_argument("--candidate_type", type=str, help="Candidate Type")
parser.add_argument("--output_path", type=int, help="Output Path")
args = parser.parse_args()


# In[ ]:


import final_utils.dataset as ds
metadata = ds.get_metadata()
gold = ds.get_gold()
queries = ds.get_queries()
precedients = ds.get_precedients()
sections = ds.get_sections()


# In[ ]:


import torch

with open(args.scores_path, "r") as f:
    scores = torch.load(f)


# In[ ]:


import torch
negs = {}
for index,query_id in enumerate(metadata[args.query_type]):
    sorted_indices = torch.argsort(scores[index], descending=True)
    sorted_candidate_ids = [metadata[args.candidate_type][j] for j in sorted_indices]
    #filtered_list = [item for item in sorted_candidate_ids if item not in gold[query_id][args.candidate_type]]
    filtered_list = sorted_candidate_ids
    negs[query_id] = filtered_list


# In[ ]:


import json 
with open(args.output_path, "w") as f:
    json.dump(negs, f)

