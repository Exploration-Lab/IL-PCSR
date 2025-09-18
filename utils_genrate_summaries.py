#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import utils.smart_summary as ss
from langchain_openai import ChatOpenAI
from langchain.output_parsers import OutputFixingParser
import tiktoken
import os
from tqdm import tqdm
import json

llm = ChatOpenAI(
    model="gpt-4o-mini",
    openai_api_key="")

encoding = tiktoken.encoding_for_model("gpt-4-turbo")


# In[ ]:


# from utils.dataset import get_metadata,get_queries,get_precedients
# metadata = get_metadata()
from utils.dataset import get_colieee_gold,get_colieee_precedents,get_colieee_queries,get_colieee_metadata

gold = get_colieee_gold()
precedents = get_colieee_precedents()
queries = get_colieee_queries()
metadata = get_colieee_metadata()


# In[ ]:


gpt_failed_folder ="smart_summarization_on_imp_parts/failed_summaries"
gpt_success_folder ="smart_summarization_on_imp_parts/colieee_summaries"

ids = metadata["train"]+metadata["test"]
documents =queries

fixing_parser = OutputFixingParser.from_llm(parser=ss.parser, llm=llm)

for id in tqdm(ids):
    filename = id+".json"
    

    if os.path.exists(os.path.join(gpt_success_folder, filename)):
        continue

    if os.path.exists(os.path.join(gpt_failed_folder, filename)):
        continue

    doc_txt = ""
    for p in  documents[id]:
        doc_txt += p + " "
    
    doc_txt = ss.clean_string(doc_txt)
    doc_tokens=  encoding.encode(doc_txt)
    doc_txt = encoding.decode(doc_tokens[:54000])

    
    try:
        out= llm.invoke(ss.query_chat_template.format_messages(document=doc_txt))

        try:
            response =ss.parser.parse(out.content)
        except Exception as e:

            try:
                print("Fixing pareser")
                response =fixing_parser.parse(out.content)
                
                
            except Exception as e:
                
                with open( os.path.join(gpt_failed_folder, filename), 'w' ) as f:
                    f.write(out.content)
            continue
        
        formatted_json_string = json.dumps(response, default=lambda o: o.__dict__ , indent=2)
    
        with open( os.path.join(gpt_success_folder, filename), 'w' ) as f:
            f.write(formatted_json_string)

    except Exception as e:
        print(e)
        continue


# In[ ]:


gpt_failed_folder ="smart_summarization_on_imp_parts/failed_summaries"
gpt_success_folder ="smart_summarization_on_imp_parts/colieee_summaries"

ids = metadata["precs"]["train"] + metadata["precs"]["test"]
documents =precedents

fixing_parser = OutputFixingParser.from_llm(parser=ss.parser1, llm=llm)

for id in tqdm(ids):
    filename = id+".json"
    

    if os.path.exists(os.path.join(gpt_success_folder, filename)):
        continue

    if os.path.exists(os.path.join(gpt_failed_folder, filename)):
        continue

    doc_txt = ""
    for p in  documents[id]:
        doc_txt += p + " "
    
    doc_txt = ss.clean_string(doc_txt)
    doc_tokens=  encoding.encode(doc_txt)
    doc_txt = encoding.decode(doc_tokens[:54000])

    
    try:
        out= llm.invoke(ss.precedent_chat_template.format_messages(document=doc_txt))

        try:
            response =ss.parser1.parse(out.content)
        except Exception as e:

            try:
                print("Fixing pareser")
                response =fixing_parser.parse(out.content)
                
                
            except Exception as e:
                
                with open( os.path.join(gpt_failed_folder, filename), 'w' ) as f:
                    f.write(out.content)
            continue
        
        formatted_json_string = json.dumps(response, default=lambda o: o.__dict__ , indent=2)
    
        with open( os.path.join(gpt_success_folder, filename), 'w' ) as f:
            f.write(formatted_json_string)

    except Exception as e:
        print(e)
        continue


# In[ ]:




