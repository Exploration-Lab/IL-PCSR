import pickle
import json
import re

# File path constants
DATASET_DIR = "dataset"
METADATA_FILE = f"{DATASET_DIR}/metadata.json"
GOLD_FILE = f"{DATASET_DIR}/gold.json"
CITATION_FILE = f"{DATASET_DIR}/citation.json"
SECTION_NAME_FILE = f"{DATASET_DIR}/section_name.json"
QUERIES_FILE = f"{DATASET_DIR}/queries.json"
SECTIONS_FILE = f"{DATASET_DIR}/sections.json"
PRECEDENTS_FILE = f"{DATASET_DIR}/precedents.json"
QUERIES_SUMMARIES_FILE = f"{DATASET_DIR}/queries_summaries.json"
QUERIES_SUMMARIES_SECS_FILE = f"{DATASET_DIR}/queries_summaries_secs.json"
PRECEDENTS_SUMMARIES_FILE = f"{DATASET_DIR}/precedents_summaries.json"


SECTION_EMBEDDINGS = f"{DATASET_DIR}/embeddings/sections_embeddings"
RR_CONSTANTS_EMBEDDINGS = f"{DATASET_DIR}/embeddings/RR_CONST_EMBD.pt"
QUERY_EMBEDDINGS_WRT_PRECS = f"{DATASET_DIR}/embeddings/s_query_embeddings_wrt_precs"
QUERY_EMBEDDINGS_WRT_SECS = f"{DATASET_DIR}/embeddings/s_query_embeddings_wrt_secs"
PRECEDENTS_SUMMARY_EMBEDDINGS = f"{DATASET_DIR}/embeddings/s_precedents_embeddings"


RR_CONSTANTS = {
    'Issue' : "Issue",
    'Section' : "Section",
    'Conclusion': "Conclusion",
    'RespArg':'Argument by Respondent',
    'PetArg': 'Argument by Petitioner',
    'Precedent':'Precedent',
    'CourtRes': "Court Reasoning",
    'Statute':"Statue",
    'CDiscource':"Court Disclosure",
    'Facts':'Facts',
    "None":"NONE",
    "NONE":"NONE",
}

EMBD_CONST = ['Argument by Petitioner',
 'Argument by Respondent',
 'Conclusion',
 'Court Disclosure',
 'Court Reasoning',
 'Facts',
 'Issue',
 'NONE',
 'Precedent',
 'Section',
 'Statue',
 'Cites',
 'Paragraph Cites']

metadata = None
gold = None
citations = None
queries = None
precedients = None
sections = None
sum_queries = None
sum_queries_for_sections = None
sum_precedients = None


def get_metadata():
    global metadata
    if metadata is None:
        with open(METADATA_FILE, "r") as f:
            
            metadata = json.load(f)
    return metadata
    
def get_gold():
    global gold
    if gold is None:
        with open(GOLD_FILE, "r") as f:
            gold = json.load(f)
    return gold

def get_citations():
    global citations
    if citations is None:
        with open(CITATION_FILE, "r") as f:
            citations = json.load(f)
    return citations

def get_section_names():
    global section_names
    if section_names is None:
        with open(SECTION_NAME_FILE, "r") as f:
            section_names = json.load(f)
    return section_names

def get_queries():
    global queries
    if queries is None:
        with open(QUERIES_FILE, "r") as f:
            queries = json.load(f)
    return queries

def get_sections():
    global sections
    if sections is None:
        with open(SECTIONS_FILE, "r") as f:
            sections = json.load(f)
    return sections

def get_precedients():
    global precedients
    if precedients is None:
        with open(PRECEDENTS_FILE, "r") as f:
            precedients = json.load(f)
    return precedients


def get_queries_summaries():
    global sum_queries
    if sum_queries is None:
        with open(QUERIES_SUMMARIES_FILE, "r") as f:
            sum_queries = json.load(f)
    return sum_queries


def get_queries_summaries_for_sections():
    global sum_queries_for_sections
    if sum_queries_for_sections is None:
        with open(QUERIES_SUMMARIES_SECS_FILE, "r") as f:
            sum_queries_for_sections = json.load(f)
    return sum_queries_for_sections


def get_precedients_summaries():
    global sum_precedients
    if sum_precedients is None:
        with open(PRECEDENTS_SUMMARIES_FILE, "r") as f:
            sum_precedients = json.load(f)
    return sum_precedients


