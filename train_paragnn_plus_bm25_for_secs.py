#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--num_cands', type=int, default=300)
parser.add_argument('--epochs', type=int, default=100)
parser.add_argument('--batch_size', type=int, default=256)
parser.add_argument('--grad_acc', type=int, default=1)
parser.add_argument('--output_dir', type=str, default="output/paragnn_plus_bm25_for_secs")
parser.add_argument('--bm25_train_scores', type=str, default=None)
parser.add_argument('--bm25_dev_scores', type=str, default=None)
parser.add_argument('--bm25_test_scores', type=str, default=None)
parser.add_argument("--trained_model", type=str, default=None, help="Path to the trained model for testing")
args = parser.parse_args(args=[])

NUM_CANDS = args.num_cands
EPOCHS = args.epochs
BATCH_SIZE = args.batch_size
GRAD_ACC = args.grad_acc
OUTPUT_DIR = args.output_dir
BM25_TRAIN_SCORES = args.bm25_train_scores
BM25_DEV_SCORES = args.bm25_dev_scores
TRAINED_MODEL = args.trained_model
CAN_TYPE = "secs"


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
import torch_geometric.transforms as T
import importlib
import dgl
from utils.score_calculator import get_gold_matrix,metrics_at_k,metrics_at_k_all
from accelerate import Accelerator


# In[ ]:


import utils.dataset as ds
metadata = ds.get_metadata()
gold = ds.get_gold()
queries = ds.get_queries()
sections = ds.get_sections()


# In[ ]:


import utils.paragnn_plus_bm25_for_secs as dgl_ut
gold_scores = get_gold_matrix(metadata["dev"],metadata["secs"],"secs")

if not os.path.exists(OUTPUT_DIR):
    os.makedirs(OUTPUT_DIR)

bm25_train_scores = torch.load(BM25_TRAIN_SCORES)
bm25_train_scores = torch.nan_to_num(bm25_train_scores, nan=0.0)
bm25_test_scores = torch.load(BM25_DEV_SCORES)

test_graph  = dgl.add_self_loop( 
    dgl_ut.GraphGenerator( metadata["dev"],metadata[CAN_TYPE] ).graph 
)


trained_model_state_dict = torch.load(TRAINED_MODEL,map_location=torch.device('cpu') )

if not os.path.exists(f"{ OUTPUT_DIR }/tokenized-sailer-allsampled.pkl"):
    print("CREATING.....")
    train_dataset = SAILERDataset(metadata['train'], gold,metadata,secs_negs, NUM_CANDS,bm25_train_scores)
    dev_dataset = []
    with open(f"{ OUTPUT_DIR }/tokenized-sailer-allsampled.pkl", 'wb') as fw:
        pkl.dump([ train_dataset, dev_dataset], fw)
else:
    print("LOADING.....")
    with open(f"{ OUTPUT_DIR }/tokenized-sailer-allsampled.pkl", 'rb') as fr:
        train_dataset, dev_dataset = pkl.load(fr)
    
print(len(train_dataset), len(dev_dataset))

sailer_collator = SAILERDataCollator(loss_type='categorical', cands_per_qry=1000)


accelerator = Accelerator(gradient_accumulation_steps=GRAD_ACC)
device = accelerator.device

train_dl = DataLoader(train_dataset, batch_size=BATCH_SIZE, collate_fn=sailer_collator, shuffle=True) 

model =  dgl_ut.CaseGnn()
test_model = dgl_ut.Test_CaseGnn()
if TRAINED_MODEL:
    model.load_state_dict( torch.load(MODEL_PATH,map_location=torch.device('cpu') ))
    print("Loaded trained model")

model.to(device)
optimizer = AdamW(model.parameters(), lr=1e-4)
num_steps = math.ceil(len(train_dataset) / (BATCH_SIZE * GRAD_ACC)) * EPOCHS
lr_scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=int(num_steps/10), num_training_steps=int(num_steps * 1.2)) 
model, optimizer, train_dl = accelerator.prepare(model, optimizer, train_dl)



# In[ ]:


LOG = []
best_MRR = 0

for epoch in range(0, EPOCHS):
    model.train()
    optimizer.zero_grad()
    
    
    total_loss = 0
    TR_loss = 0
    
    for k,batch in enumerate(tqdm(train_dl, desc="Training...")):
        torch.cuda.empty_cache()
        model.train()

        # if random.random() < 0.5:  # 10% probability
        #     continue  # Skip this batch
        
        dict_on_device = {key: value.to(device) for key, value in batch.items()}
    
        with accelerator.accumulate(model):
            try:
                loss = model(**dict_on_device)
            except torch.cuda.OutOfMemoryError:
                print("skipped")
                continue

            try:
                accelerator.backward(loss)
            except torch.cuda.OutOfMemoryError:
                print("skipped")
                del loss
                continue

            # loss = model(**dict_on_device)
            # accelerator.backward(loss)

            optimizer.step()
            lr_scheduler.step()
            optimizer.zero_grad()
            total_loss += loss.item()

    TR_loss = total_loss/len(train_dl)
    
    
    if (epoch+1) % 1 == 0: 
        model.to('cpu')
        test_model.load_state_dict( model.state_dict())
        test_model.eval()
        with torch.no_grad():
            prec_score,alphas = test_model(bm25_test_scores,test_graph)
        PP, RR, FF, MRR_min=metrics_at_k(test_gold_Scores, prec_score,top_k=10)
        MRR_min = max(FF)
        LOG.append( (epoch, TR_loss, MRR_min) )
        
         
        if MRR_min > best_MRR:
            best_MRR = MRR_min
            LOG.append( ("Saving model at",epoch, TR_loss, MRR_min) )
            torch.save(model.state_dict(), f"{ OUTPUT_DIR }/pytorch_model.bin")
            # torch.save(optimizer.state_dict(), f"{ OUTPUT_DIR }/optimizer.pt")
            # torch.save(lr_scheduler.state_dict(), f"{ OUTPUT_DIR }/scheduler.pt")
        else:
            print(f"Epoch {epoch + 1}: Total Loss = {TR_loss} , MRR_min = {MRR_min} ") 

            
        model.to(device)   

    with open(f"{ OUTPUT_DIR}/log.json", 'w') as fw:
        json.dump(LOG, fw, indent=4) 

