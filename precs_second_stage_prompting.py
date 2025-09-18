#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import torch
import os
from utils.dataset import get_gold,get_metadata,get_precedients,get_queries
from utils.score_calculator import get_gold_matrix
import utils.smart_summary as ss
from langchain.chat_models import init_chat_model
from utils.genral import get_rank_list
import json
import tiktoken
from utils.score_calculator import get_gold_matrix,metrics_at_k_all
import random
from tqdm import tqdm


openai_api_key="ADD_YOUR_KEY_HERE"
encoding = tiktoken.encoding_for_model("gpt-4") 
if not os.environ.get("OPENAI_API_KEY"):
  os.environ["OPENAI_API_KEY"] = openai_api_key
llm = init_chat_model(
    #model="gpt-4o-mini",
    # model="gpt-4o",
    #model="o3",
    #model="o3-mini",
    model="gpt-4.1",
    temperature=1, 
)


metadata = get_metadata()
gold = get_gold()

precedents = get_precedients()
queries = get_queries()
citations = json.load(open("dataset/lsipcr-precs-gold-all.json"))

q_ids =metadata["test"]
p_ids =metadata["precs"]
prec_score  = torch.load(f"ARR_MAY/scores/pipeline_precs.pt")
test_gold_scores = torch.load("dataset/scores/test_precs_scores.pt")



precs_rank_list = get_rank_list(q_ids,p_ids,prec_score)
secs_names = json.load(open("dataset/lsipcr-secnames.json"))

#pari_score_path = "dataset/genai/precs/without.json"
pari_score_path = "ARR_MAY/gen_ai/precs/with_secs.json"


query_llm_secs = json.load(open("ARR_MAY/gen_ai/secs/without.json"))


# In[ ]:


#CALCULATE SCORES FOR ONLY PRECEDENTS AND QUERIES

scores = json.load(open(pari_score_path))
for id in tqdm( q_ids):
    query_text = "\n".join(p[1] for p in queries[id])
    query_text = encoding.decode( encoding.encode(query_text)[:10000] )  

    query_secs = query_llm_secs[id]
    query_secs_names = [secs_names[s] for s in query_secs]

    
    for i, p_id in enumerate(precs_rank_list[id][:10]):
        if f"{id}_{p_id}" in scores:
            #print(f"Already processed {id} with precedent {p_id}")
            continue

        p_secs = [ secs_names[s_id] for s_id in citations[p_id]["secs"] if s_id in metadata["secs"] ]
        
        prec_text = "\n".join(p[1] for p in precedents[p_id])
        prec_text = encoding.decode( encoding.encode(prec_text)[:10000] )  
        
        try:
            out = llm.invoke(ss.precs_secs_chat_template.format_messages(query_txt=query_text, query_secs= json.dumps(query_secs_names),
                                                                         precs_txt=prec_text,
                                                                          precs_secs= json.dumps(p_secs)))
            score = ss.bool_parser.parse(out.content)
                

            scores[f"{id}_{p_id}"] = score
            with open(pari_score_path, "w") as f:
                json.dump(scores, f)
        
        except Exception as e:
            print(f"Error processing {id} with precedent {p_id}: {e}")


# In[ ]:


#GENERATE THE SCORES
scores = json.load(open(pari_score_path))
#scores = json.load(open("dataset/genai/precs/without.json"))

llm_only_precs_scores = torch.zeros(len(q_ids),len(p_ids))
for q_id in q_ids:

    q_index = q_ids.index(q_id)    
    
    for p_id in precs_rank_list[q_id][:10]:
        p_index = p_ids.index(p_id)

        if scores[f"{q_id}_{p_id}"]:
            llm_only_precs_scores[q_index][p_index] = 1
        

torch.save(llm_only_precs_scores,"scores/gen_ai/precs_with_secs.pt")

