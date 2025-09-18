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

openai_api_key="ADD_YOUR_KEY_HERE"
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
#queries_summaries_for_sections = get_queries_summaries_for_sections()

q_ids =metadata["test"]
s_ids =metadata["secs"]

sec_score  = torch.load(f"ARR_MAY/scores/pipeline_secs.pt")
secs_rank_list = get_rank_list(q_ids,s_ids,sec_score)
test_gold_scores = torch.load("dataset/scores/test_secs_scores.pt")

secs_names = json.load(open("dataset/lsipcr-secnames.json"))

#indices = [i for i, qid in enumerate(q_ids) if qid in ids_filtered_precs]


# In[ ]:


json_file = "ARR_MAY/gen_ai/secs/without.json"
without_preds = json.load(open(json_file))
topk = 20
for id in tqdm(q_ids):
    
    if id in without_preds:
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

    try:
        prompt = ss.classification_template.format_messages(query=query_txt,statute=json.dumps(section_input))
        out= llm.invoke( prompt )
        response = ss.classification_parser.parse(out.content)
        stats = []
        for s in response.statutes:
            if s in secs_rank_list[id][:topk]:
                stats.append(s)
            # if s not in secs_rank_list[id][:topk]:
            #     raise Exception(f"Section {s} not in topk list for id {id}")
        without_preds[id] = stats
        with open(json_file, "w") as f:
            json.dump(without_preds, f, indent=4)
        
    except Exception as e:
        print(e)

    


# In[ ]:


with_preds =  json.load(open("ARR_MAY/gen_ai/secs/with_precs.json"))
#without_preds = json.load(open("ARR_MAY/gen_ai/secs/without.json"))
#without_preds = json.load(open("dataset/genai/secs/without.json"))

preds = with_preds

llm_sec_scores = torch.zeros(len(q_ids),len(s_ids))
for q_id in q_ids:

    q_index = q_ids.index(q_id)    
    
    for s_id in preds[q_id]:
        s_index = s_ids.index(s_id)

        llm_sec_scores[q_index][s_index] = 1

#torch.save(llm_sec_scores,"scores/gen_ai/secs_without_precs.pt")


# In[ ]:


# s1 = torch.load("ARR_MAY/scores/only_secs.pt")
# s2 = torch.load("ARR_MAY/scores/multitask_secs.pt")
llm_sec_score = llm_sec_scores

final_score = sec_score.clone()
final_score[llm_sec_score == 1] += torch.max(sec_score).item()

rs =metrics_at_k_all(test_gold_scores, final_score)
print(rs["map"])
print(rs["mrr"])
print(max(rs["mF"]))


# In[ ]:


#torch.save(llm_sec_scores,"scores/gen_ai/secs_without_precs.pt")

