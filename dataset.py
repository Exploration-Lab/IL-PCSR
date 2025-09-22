#!/usr/bin/env python
# coding: utf-8

# In[ ]:


from datasets import load_dataset
dataset = load_dataset("Exploration-Lab/IL-PCSR", "queries")
statute_dataset = load_dataset("Exploration-Lab/IL-PCSR", "statutes")
precedent_dataset = load_dataset("Exploration-Lab/IL-PCSR", "precedents")


# In[ ]:


metadata = {}
metadata["train"] = list(dataset["train_queries"]["id"])
metadata["dev"]= list(dataset["dev_queries"]["id"])
metadata["test"] = list(dataset["test_queries"]["id"])
metadata["secs"] = list(statute_dataset["statute_candidates"]["id"])
metadata["precs"] = list(precedent_dataset["precedent_candidates"]["id"])

gold ={}

queries = {}
for split in ["train_queries", "dev_queries", "test_queries"]:
    for row in dataset[split]:
        qid = row["id"]
        texts = row["text"]
        rrs = row["rr"]
        pairs = [[rr, txt] for rr, txt in zip(rrs, texts)]
        queries[qid] = pairs

        gold[qid] = {"precs": row["relevant_precedent_ids"], "secs": row["relevant_statute_ids"]}

precedents={}
for row in precedent_dataset['precedent_candidates']:
    qid = row["id"]
    texts = row["text"]
    rrs = row["rhetorical_roles"]
    pairs = [[rr, txt] for rr, txt in zip(rrs, texts)]
    precedents[qid] = pairs

statutes={}
for row in statute_dataset['statute_candidates"]:
    qid = row["id"]
    texts = row["text"]
    pairs = [[None, txt] for txt in texts]
    statutes[qid] = pairs


# In[ ]:


import utils.dataset as ds 
import json
with open(ds.METADATA_FILE,"w") as f:
    json.dump(metadata,f)

with open(ds.GOLD_FILE,"w") as f:
    json.dump(gold,f)

with open(ds.QUERIES_FILE,"w") as f:
    json.dump(queries,f)

with open(ds.PRECEDENTS_FILE,"w") as f:
    json.dump(precedents,f)

with open(ds.SECTIONS_FILE,"w") as f:
    json.dump(statutes,f)

