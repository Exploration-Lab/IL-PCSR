from utils.dataset import get_precedients_summaries,get_queries_summaries,get_sections,get_queries_summaries_for_sections,EMBD_CONST
import torch
import tqdm

from itertools import product
import random
import pickle
import dgl
import time
import json
import dgl.function as fn
import torch.nn.functional as F
import torch.nn as nn
from utils.EUGATConv import EUGATConv
from dgl.nn import EGATConv
import torch
import math
import torch as th


QUERY_EMBD_PATH = "dataset/section_embeddings/"
text2embd= pickle.load(open("dataset/sections_text2embd.pkl","rb"))

QUERY1_EMBD_PATH = "dataset/embeddings/"
text2embd1= pickle.load(open("dataset/text2embd.pkl","rb"))

CANDIDATE_EMBD_PATH = "dataset/embeddings/"
EMBD_CONST_PT =torch.load(f"dataset/embeddings/EMBD_CONST.pt").to("cpu")



queries = get_queries_summaries_for_sections()
queries1 = get_queries_summaries()

candidates = get_sections()




class GraphGenerator:
    def __init__(self,query_index2id, candidate_index2id):
        self.query_index2id = query_index2id
        self.candidate_index2id = candidate_index2id 
        self.index2id = query_index2id + candidate_index2id

        self.u_ids = []
        self.v_ids = []
        self.edge_features = torch.empty( (0,768) )
        self.query_para_count = 0
        self.candidate_para_count = 0
        self.add_edge_features()

        self.n_data = torch.empty( (len(self.index2id)+self.edge_features.shape[0],768) )
        self.add_doc_to_entity_connection()


        #GENERATE GRAPH
        self.graph = dgl.graph( (self.u_ids , self.v_ids), num_nodes=self.n_data.shape[0] )
        self.graph.ndata["h"] = self.n_data

        query_mask = torch.zeros(self.n_data.shape[0], dtype=torch.float32)
        query_mask[:len(self.query_index2id)] = 1.0
        self.graph.ndata["query_mask"] = query_mask
        
        

        candidate_mask = torch.zeros(self.n_data.shape[0], dtype=torch.float32)
        candidate_mask[len(self.query_index2id): len(self.index2id)] = 1.0
        self.graph.ndata["candidate_mask"] = candidate_mask

        query_para_mask = torch.zeros(self.n_data.shape[0], dtype=torch.float32)
        query_para_mask[len(self.index2id):len(self.index2id)+self.query_para_count] = 1.0
        self.graph.ndata["query_para_mask"] = query_para_mask

        candidate_para_mask = torch.zeros(self.n_data.shape[0], dtype=torch.float32)
        candidate_para_mask[len(self.index2id)+ self.query_para_count:] = 1.0
        self.graph.ndata["candidate_para_mask"] = candidate_para_mask


        self.graph.edata["h"] = self.edge_features
    
    def add_edge_features(self):
        edge_strings = []
        

        for index,id in  enumerate(  self.index2id ) :

            if index < len(self.query_index2id):
                titles = [o[0] for o in queries[id]]
                titles1 = [o[0] for o in queries1[id]]
                titles.extend(titles1)
                
                self.query_para_count += len(titles)
                edge_strings.extend( titles )
            else:
                paragraphs = candidates[id]
                self.candidate_para_count += len(paragraphs)


                for i, paragraph in enumerate(paragraphs):
                        edge_strings.append("NONE")

            
                

        self.edge_features = torch.empty( (len(edge_strings),768) )             
        for i,edge_f in enumerate(edge_strings):
            if edge_f in ["Cites","NONE"]:
                self.edge_features[i] = EMBD_CONST_PT[EMBD_CONST.index(edge_f)]
            elif edge_f in text2embd:
                self.edge_features[i] = text2embd[edge_f]
            else:
                self.edge_features[i] = text2embd1[edge_f]
            

        

    def add_doc_to_entity_connection(self):
        
        offset = len(self.index2id)

        for index,id in  enumerate(  self.index2id ) :

            if index < len(self.query_index2id):
                sent_embds = torch.load(QUERY_EMBD_PATH+id+".pt").to("cpu")
                sent1_embds = torch.load(QUERY1_EMBD_PATH+id+".pt").to("cpu")
                sent_embds = torch.cat([sent_embds,sent1_embds],dim=0)
            else:
                sent_embds = torch.load(CANDIDATE_EMBD_PATH+id+".pt").to("cpu")
            
            
            self.n_data[index] = torch.mean(sent_embds,dim=0)

            end_idx = offset + sent_embds.size(0)

            self.n_data[offset:end_idx, :] = sent_embds


            
            for i in range( sent_embds.size(0) ):
                self.u_ids.append( offset + i )
                self.v_ids.append( index)

            offset += sent_embds.size(0)


        

class SAILERDataset(Dataset):
    def __init__(self, ids,  gold,metadata,secs_negs, NUM_CANDS,bm25_train_scores):
        self.dataset = []
        
        print("Creating dataset........")
        
        for id in ids:
            data = {'id': id}
            rel_stats = gold[id]['secs']
            rel_precs = gold[id]['precs']
            #data['prec_all_relevant'] = rel_precs
            data['prec_all_relevant'] = rel_stats
            q_index = ids.index(id)

            #for s,p in product(rel_stats, rel_precs):
            for s,p in product(rel_precs, rel_stats):
                data2 = deepcopy(data)
                data2['prec_relevant'] = p

                # filtered_precs_with_indices = [
                #     (index, value) for index, value in enumerate(metadata[CAN_TYPE]) if value not in rel_precs
                # ]   
                
                filtered_precs_with_indices = [
                    (index, value) for index, value in enumerate(secs_negs[id][:400]) if value not in rel_precs
                ]   
                sampled_indices, sampled_ids = zip(* random.sample( filtered_precs_with_indices, NUM_CANDS-1 ) )
                sampled_indices = list(sampled_indices)
                sampled_indices.insert(0,metadata["secs"].index(p))

                data2['prec_bm25_negatives'] =list(sampled_ids)
                data2["bm25_scores"]=bm25_train_scores[q_index][sampled_indices]
                
                self.dataset.append(data2)
    
    def __len__(self):
        return len(self.dataset)
    
    def __getitem__(self, idx):
        return self.dataset[idx]  


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
        
        graph_generator = GraphGenerator(index2id['query'], index2id['candidate'])
        batch['graph'] =dgl.add_self_loop( graph_generator.graph)
        
        return batch

    

        