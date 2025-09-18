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


queries = ds.get_queries_summaries()


for q_id in tqdm(metadata["train"]+metadata["dev"]+metadata["test"]):
    q_paras = queries[q_id]
    paras = [para[1] for para in q_paras]
    with torch.no_grad():
        embeddings = model.encode(paras, convert_to_tensor=True)
    torch.save(embeddings, f"{ds.QUERY_EMBEDDINGS_WRT_PRECS}/{q_id}.pt")


# In[ ]:


queries = ds.get_queries_summaries_for_sections()


for q_id in tqdm(metadata["train"]+metadata["dev"]+metadata["test"]):
    q_paras = queries[q_id]
    paras = [para[1] for para in q_paras]
    with torch.no_grad():
        embeddings = model.encode(paras, convert_to_tensor=True)
    torch.save(embeddings, f"{ds.QUERY_EMBEDDINGS_WRT_SECS}/{q_id}.pt")

