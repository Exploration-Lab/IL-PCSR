#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import os
import torch

os.environ['CUDA_VISIBLE_DEVICES'] = '1'

# Check if GPU is available
if torch.cuda.is_available():
    # Get the number of available GPUs
    num_gpus = torch.cuda.device_count()
    print(f"Number of available GPUs: {num_gpus}")
    for gpu_id in range(num_gpus):
        print(f"GPU {gpu_id}: {torch.cuda.get_device_name(gpu_id)}")
else:
    print("No GPU available.")


# In[ ]:


from sentence_transformers import SentenceTransformer, models
import torch
import utils.dataset as ds
model_name = 'sentence-transformers/all-mpnet-base-v2'
model = SentenceTransformer(model_name)


# In[ ]:


from utils.dataset import EMBD_CONST
with torch.no_grad():
    embeddings = model.encode(EMBD_CONST, convert_to_tensor=True)
torch.save(embeddings, ds.RR_CONSTANTS_EMBEDDINGS)


# In[ ]:


import utils.dataset as ds
from tqdm import tqdm
import json
metadata = ds.get_metadata()


sections = ds.get_sections()


for s_id in tqdm(metadata["secs"]):
    s_paras = sections[s_id]
    paras = [para[1] for para in s_paras]
    with torch.no_grad():
        embeddings = model.encode(paras, convert_to_tensor=True)
    torch.save(embeddings, f"{ds.SECTION_EMBEDDINGS}/{s_id}.pt")

