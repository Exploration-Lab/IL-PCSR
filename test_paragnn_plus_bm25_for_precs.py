#!/usr/bin/env python
# coding: utf-8

# disconneted graph while training and testing , same as sailer or some embedding genrateor model

# In[ ]:


import os
import torch

# Check if MPS is available
if torch.backends.mps.is_available():
    print("✅ MPS is available.")
    print("Is MPS built: ", torch.backends.mps.is_built())
    print("Current device: MPS")
else:
    print("❌ MPS is not available.")


# In[ ]:


import os
import json
import pickle as pkl
from dataclasses import dataclass
import torch
import random
import math
import importlib
from copy import deepcopy
from itertools import product
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
import pickle 
from transformers import AdamW
from transformers import get_linear_schedule_with_warmup
from transformers.data.data_collator import DataCollatorMixin
from utils.dataset import get_gold,get_metadata,get_precedients,get_queries,get_precs_negs
import utils.dgl_ensemble_summaries as dgl_ut
import torch_geometric.transforms as T
import importlib
import dgl
from utils.score_calculator import get_gold_matrix,metrics_at_k,metrics_at_k_all
importlib.reload(dgl_ut)

NUM_CANDS = 900
EPOCHS = 100
BATCH_SIZE = 256
GRAD_ACC = 1
OUTPUT_DIR = "output/pipeline_precs"
VERIFICATION_ON = "test"
CAN_TYPE = "precs"


if not os.path.exists(OUTPUT_DIR):
    os.makedirs(OUTPUT_DIR)


metadata = get_metadata()
gold = get_gold()
negs =get_precs_negs()

bm25_train_scores = torch.load("scores/train_bm25n5norm.pt")
bm25_test_scores = torch.load("scores/bm25n5norm.pt")

test_graph  = dgl.add_self_loop( 
    dgl_ut.GraphGenerator( metadata["test"],metadata[CAN_TYPE] ).graph 
    )

SCORE_PATH = f"dataset/scores/{VERIFICATION_ON}_{CAN_TYPE}_scores.pt"
if not os.path.exists(SCORE_PATH):
    test_gold_Scores = get_gold_matrix(metadata[VERIFICATION_ON],metadata[CAN_TYPE],CAN_TYPE)
    torch.save(test_gold_Scores, SCORE_PATH)
else:
    test_gold_Scores = torch.load(SCORE_PATH)


# In[ ]:


#test
test_model = dgl_ut.Test_CaseGnn()
test_model.load_state_dict( torch.load("output/setting1_ensemble_multitask_summaries/precs_pytorch_model.bin",map_location=torch.device('cpu') ), strict=False)
test_model.eval()
prec_score,alphas = test_model(bm25_test_scores,test_graph)
rs =metrics_at_k_all(test_gold_Scores, prec_score)

print(rs["map"])
print(rs["mrr"])
print(max(rs["mF"]))
print(alphas.mean())

