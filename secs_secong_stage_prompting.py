#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import torch
import os
from utils.dataset import get_gold,get_metadata,get_sections,get_queries,get_queries_summaries_for_sections
from utils.score_calculator import get_gold_matrix,metrics_at_k_all
import utils.smart_summary_secs as ss
from langchain.chat_models import init_chat_model
import json
import tiktoken
import random
from utils.genral import get_rank_list
from tqdm import tqdm


openai_api_key="ADD_YOUR_KEY"
encoding = tiktoken.encoding_for_model("gpt-4") 
if not os.environ.get("OPENAI_API_KEY"):
  os.environ["OPENAI_API_KEY"] = openai_api_key
llm = init_chat_model(
    #model="gpt-4o-mini",
    # model="gpt-4o",
    #model="o3",
    model="gpt-4.1",
    #model="o3-mini",
    temperature=1, 
)

metadata = get_metadata()
gold = get_gold()
sections = get_sections()
queries = get_queries()
citations = json.load(open("dataset/lsipcr-precs-gold-all.json"))
#queries_summaries_for_sections = get_queries_summaries_for_sections()

q_ids =metadata["test"]
s_ids =metadata["secs"]
p_ids =metadata["precs"]

sec_score  = torch.load(f"ARR_MAY/scores/pipeline_secs.pt")
prec_score  = torch.load(f"ARR_MAY/scores/pipeline_precs.pt")
secs_rank_list = get_rank_list(q_ids,s_ids,sec_score)
precs_rank_list = get_rank_list(q_ids,p_ids,prec_score)


test_gold_scores = torch.load("dataset/scores/test_secs_scores.pt")

secs_names = json.load(open("dataset/lsipcr-secnames.json"))
#pair_score_path ="ARR_MAY/gen_ai/secs/with_precs.json"
pair_score_path ="ARR_MAY/gen_ai/secs/with_precs_after_secs.json"
#pair_score_path = "ARR_MAY/gen_ai/secs/without.json"


# In[ ]:


query_precs_path = "ARR_MAY/gen_ai/precs/with_secs.json"
#query_precs_path = "ARR_MAY/gen_ai/precs/without.json"
query_precs_mapping = json.load(open(query_precs_path))
query_predicted_precs = {}
for id in tqdm(q_ids):
    query_predicted_precs[id] = []
    for p_id in precs_rank_list[id][:10]:
        if query_precs_mapping[f"{id}_{p_id}"]:
            query_predicted_precs[id].append(p_id)


# In[ ]:


scores = json.load(open(pair_score_path))
topk = 20
for id in tqdm(q_ids):
    
    if id in scores:
        #print("Already exists: ", id)
        continue
    

    q_txt = " \n ".join(p[1] for p in queries[id])
    q_tokens = encoding.encode(q_txt)
    query_txt = encoding.decode( q_tokens[:10000] )
    #print("ID: ",id, "Tokens",len(q_tokens))

    section_input = []

    for s_id in secs_rank_list[id][:topk]:
        s = {}
        s["id"] = s_id
        s["name"] = secs_names[s_id]
        section_input.append(s)

    section_input_from_precs = []
    for p_id in query_predicted_precs[id]:
        p_secs  = [ s_id for s_id in citations[p_id]["secs"] if s_id in metadata["secs"]]
        for s_id in p_secs:
            if s_id not in secs_rank_list[id][:topk]:
                s = {}
                s["id"] = s_id
                s["name"] = secs_names[s_id]
                section_input_from_precs.append(s)

    try:
        prompt = ss.classification_template.format_messages(query=query_txt,statute=json.dumps(section_input+section_input_from_precs),query_id=id)
        out= llm.invoke( prompt )
        response = ss.classification_parser.parse(out.content)
        preds_stats =[]
        for s in response.statutes:
            #if s not in secs_rank_list[id][:topk]:
            # if s not in metadata["secs"]:
            #     raise Exception(f"Section {s} not in topk list for id {id}")
            if s in metadata["secs"]:
                preds_stats.append(s)
        
        scores[id] = preds_stats
        with open(pair_score_path, "w") as f:
            json.dump(scores, f, indent=4)
    except Exception as e:
        print(e)


# In[ ]:


scores = json.load(open(pair_score_path))
llm_sec_scores = torch.zeros(len(q_ids),len(s_ids))
for q_id in q_ids:

    q_index = q_ids.index(q_id)    
    
    for s_id in scores[q_id]:
        s_index = s_ids.index(s_id)

        llm_sec_scores[q_index][s_index] = 1

#torch.save(llm_sec_scores,"scores/gen_ai/secs_without_precs.pt")


# In[ ]:


#indices = [i for i, qid in enumerate(q_ids) if qid in ids_filtered_precs]

#llm_sec_score = torch.load("scores/llm_secs_scores.pt")


final_score = sec_score.clone()
final_score[llm_sec_scores == 1] += torch.max(sec_score).item()

rs =metrics_at_k_all(test_gold_scores, final_score)
print(rs["map"])
print(rs["mrr"])
print(max(rs["mF"]))

