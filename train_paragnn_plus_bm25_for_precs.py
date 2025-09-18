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
#test_model.load_state_dict( torch.load("output/setting1_ensemble_multitask_summaries/precs_pytorch_model.bin",map_location=torch.device('cpu') ), strict=False)
test_model.eval()
prec_score,alphas = test_model(bm25_test_scores,test_graph)
rs =metrics_at_k_all(test_gold_Scores, prec_score)

print(rs["map"])
print(rs["mrr"])
print(max(rs["mF"]))
print(alphas.mean())


# In[ ]:


class SAILERDataset(Dataset):
    def __init__(self, ids,  gold=None, bm25neg_citations=None, num_neg_samples=1000):
        self.dataset = []
        
        print("Creating dataset........")
        
        for id in ids:
            data = {'id': id}
            rel_stats = gold[id]['secs']
            rel_precs = gold[id]['precs']
            data['prec_all_relevant'] = rel_precs
            #data['prec_all_relevant'] = rel_stats
            q_index = ids.index(id)

            for s,p in product(rel_stats, rel_precs):
            #for s,p in product(rel_precs, rel_stats):
                data2 = deepcopy(data)
                data2['prec_relevant'] = p

                # filtered_precs_with_indices = [
                #     (index, value) for index, value in enumerate(metadata[CAN_TYPE]) if value not in rel_precs
                # ]   
                
                filtered_precs_with_indices = [
                    (index, value) for index, value in enumerate(negs[id][:1500]) if value not in rel_precs
                ]   
                sampled_indices, sampled_ids = zip(* random.sample( filtered_precs_with_indices, NUM_CANDS-1 ) )
                sampled_indices = list(sampled_indices)
                sampled_indices.insert(0,metadata[CAN_TYPE].index(p))

                data2['prec_bm25_negatives'] =list(sampled_ids)
                data2["bm25_scores"]=bm25_train_scores[q_index][sampled_indices]
                
                self.dataset.append(data2)
    
    def __len__(self):
        return len(self.dataset)
    
    def __getitem__(self, idx):
        return self.dataset[idx]
    


if not os.path.exists(f"{ OUTPUT_DIR }/tokenized-sailer-allsampled.pkl"):
    print("CREATING.....")
    train_dataset = SAILERDataset(metadata['train'], gold=gold)
    dev_dataset = []
    with open(f"{ OUTPUT_DIR }/tokenized-sailer-allsampled.pkl", 'wb') as fw:
        pkl.dump([ train_dataset, dev_dataset], fw)
else:
    print("LOADING.....")
    with open(f"{ OUTPUT_DIR }/tokenized-sailer-allsampled.pkl", 'rb') as fr:
        train_dataset, dev_dataset = pkl.load(fr)
    
print(len(train_dataset), len(dev_dataset))


# In[ ]:


import dgl
# import utils.dgl_ensemble_triplet as dgl_ut
# importlib.reload(dgl_ut)

@dataclass
class SAILERDataCollator(DataCollatorMixin):
    def __init__(self , loss_type='categorical', cands_per_qry=8):
        self.cands_per_qry = cands_per_qry
        self.loss_type = loss_type
           
    def __call__(self, queries):
        batch = {}
        for k in ['query_pos','candidate_pos','bm25_scores']:
            batch[k] = []
        
        index2id = {}
        index2id['candidate'] = []
        index2id['query'] = []
        
        for qry in queries:
            if qry['id'] not in index2id['query']:
                index2id['query'].append(qry['id'])
            batch['query_pos'].append( index2id['query'].index(qry['id']) )
            
            ppos = []
            for i,p in enumerate( [qry['prec_relevant']] + qry['prec_bm25_negatives'] ):

                if p not in index2id['candidate']: 
                    index2id['candidate'].append(p)

                ppos.append(index2id['candidate'].index(p))
            batch['candidate_pos'].append(ppos)
            batch['bm25_scores'].append(qry['bm25_scores'])



        #---------------------------------DOC_POS---------------------------------
        batch['query_pos'] = torch.tensor(batch['query_pos'])
        batch['candidate_pos'] = torch.tensor(batch['candidate_pos'])
        

        #---------------------------------CAN_RELI_LABEL---------------------------------
        batch['candidate_relevance_labels'] = torch.zeros_like(batch['candidate_pos'])
        batch['candidate_relevance_labels'][:, 0] = 1

        batch['bm25_scores'] = torch.stack(batch['bm25_scores'])



        #---------------------------------GRAPH---------------------------------
        
        graph_generator = dgl_ut.GraphGenerator(index2id['query'], index2id['candidate'])
        batch['graph'] =dgl.add_self_loop( graph_generator.graph)
        
        return batch
    
sailer_collator = SAILERDataCollator(loss_type='categorical', cands_per_qry=1000)


# In[ ]:


sample_batch = sailer_collator(train_dataset[0:100])
print(sample_batch.keys())


# In[ ]:


from accelerate import Accelerator
accelerator = Accelerator(gradient_accumulation_steps=GRAD_ACC)
device = accelerator.device

train_dl = DataLoader(train_dataset, batch_size=BATCH_SIZE, collate_fn=sailer_collator, shuffle=True) 


model =  dgl_ut.CaseGnn()
# for param in model.eugat_gnn.parameters():
#     param.requires_grad = False
test_model = dgl_ut.Test_CaseGnn()
#model.load_state_dict( torch.load("output/setting1_triplets_secs/pytorch_model.bin",map_location=torch.device('cpu') ), strict=False)
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

