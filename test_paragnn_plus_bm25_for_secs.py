#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--output_dir', type=str, default="output/a_pipeline_secs")
parser.add_argument('--bm25_test_scores', type=str, default=None)
parser.add_argument("--trained_model", type=str, default=None, help="Path to the trained model for testing")
args = parser.parse_args(args=[])


OUTPUT_DIR = args.output_dir
BM25_TEST_SCORES = args.bm25_test_scores
CAN_TYPE = "secs"
TRAINED_MODEL = args.trained_model


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
from utils.dataset import get_gold,get_metadata,get_precedients,get_queries,get_secs_negs
import utils.dgl_ensemble_summaries_secs as dgl_ut
import torch_geometric.transforms as T
import importlib
import dgl
from utils.score_calculator import get_gold_matrix,metrics_at_k,metrics_at_k_all
from accelerate import Accelerator


# In[ ]:


import final_utils.dataset as ds
metadata = ds.get_metadata()
gold = ds.get_gold()
queries = ds.get_queries()
precedients = ds.get_precedients()
sections = ds.get_sections()


# In[ ]:


gold_scores =get_gold_matrix(metadata["test"],metadata["secs"],"secs")

new_state_dict = torch.load(MODEL_PATH,map_location=torch.device('cpu') )
test_model = dgl_ut.Test_CaseGnn()
test_model.load_state_dict( new_state_dict, strict=False)
test_model.eval()
test_model.to("cpu")
prec_score,alphas = test_model(bm25_test_scores,test_graph)
rs =metrics_at_k_all(test_gold_Scores, prec_score)

print(rs["map"])
print(rs["mrr"])
print(max(rs["mF"]))
print(alphas.mean())

