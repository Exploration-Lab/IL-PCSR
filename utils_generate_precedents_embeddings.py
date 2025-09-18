#!/usr/bin/env python
# coding: utf-8

# In[ ]:


from sentence_transformers import SentenceTransformer, models
import torch
import utils.dataset as ds
model_name = 'sentence-transformers/all-mpnet-base-v2'
model = SentenceTransformer(model_name)


# In[ ]:


import utils.dataset as ds
from tqdm import tqdm
import json
metadata = ds.get_metadata()


precedents = ds.get_precedients_summaries()


for q_id in tqdm(metadata["precs"]):
    q_paras = precedents[q_id]
    paras = [para[1] for para in q_paras]
    with torch.no_grad():
        embeddings = model.encode(paras, convert_to_tensor=True)
    torch.save(embeddings, f"{ds.PRECEDENTS_SUMMARY_EMBEDDINGS}/{q_id}.pt")

