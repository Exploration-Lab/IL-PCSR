from utils.dataset import get_precedients_summaries,get_queries_summaries,get_queries_summaries_for_sections
import torch
import tqdm

from itertools import product
import random
import pickle
import dgl
import time
import json

EMBD_PATH = "smart_summarization_on_imp_parts/embeddings/"

#text2embd= pickle.load(open("smart_summarization_on_imp_parts/text2embd.pkl","rb"))
import torch

text2embd = torch.load("smart_summarization_on_imp_parts/text2embd.pkl", map_location=torch.device('cpu'),weights_only=True)

EMBD_PATH1 = "smart_summarization_on_imp_parts/section_embeddings/"
text2embd1= pickle.load(open("smart_summarization_on_imp_parts/sections_text2embd.pkl","rb"))



queries = get_queries_summaries()
queries1 = get_queries_summaries_for_sections()

candidates = get_precedients_summaries()




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
                titles = [o[0] for o in candidates[id]]
                
                self.candidate_para_count += len(titles)
                edge_strings.extend( titles )

            
                

        self.edge_features = torch.empty( (len(edge_strings),768) )             
        for i,edge_f in enumerate(edge_strings):
            if edge_f in text2embd:
                self.edge_features[i] = text2embd[edge_f]
            else:
                self.edge_features[i] = text2embd1[edge_f]

        

    def add_doc_to_entity_connection(self):
        
        offset = len(self.index2id)

        for index,id in  enumerate(  self.index2id ) :

            if index < len(self.query_index2id):
                sent_embds = torch.load(EMBD_PATH+id+".pt").to("cpu")
                sent1_embds = torch.load(EMBD_PATH1+id+".pt").to("cpu")
                sent_embds = torch.cat([sent_embds,sent1_embds],dim=0)
            else:
                sent_embds = torch.load(EMBD_PATH+id+".pt").to("cpu")
            
            
            self.n_data[index] = torch.mean(sent_embds,dim=0)

            end_idx = offset + sent_embds.size(0)

            self.n_data[offset:end_idx, :] = sent_embds


            
            for i in range( sent_embds.size(0) ):
                self.u_ids.append( offset + i )
                self.v_ids.append( index)

            offset += sent_embds.size(0)


        





import dgl.function as fn
import torch.nn.functional as F
import torch.nn as nn
from utils.EUGATConv import EUGATConv
from dgl.nn import EGATConv
import torch
import math
import torch as th

   

class EUGATGNN(nn.Module):
    def __init__(self, in_dim, h_dim, out_dim, dropout, num_head):
        super(EUGATGNN, self).__init__()
        self.hidden_size = h_dim
        self.in_dim = in_dim
        self.EUGATConv1 = EUGATConv(in_feats=in_dim, edge_feats=in_dim, out_feats=out_dim, out_edge_feats=out_dim, num_heads=num_head,allow_zero_in_degree=True)
        self.EUGATConv2 = EUGATConv(in_feats=in_dim, edge_feats=in_dim, out_feats=out_dim, out_edge_feats=out_dim, num_heads=num_head,allow_zero_in_degree=True)
        self.embedding_dropout1 = nn.Dropout(dropout)
        self.embedding_dropout2 = nn.Dropout(dropout)
        self.reset_parameters()

    def reset_parameters(self):
        if self.hidden_size == 0:
            stdv = 1.0 / math.sqrt(self.in_dim)
            for weight in self.parameters():
                weight.data.uniform_(-stdv, stdv)
        else:
            stdv = 1.0 / math.sqrt(self.hidden_size)
            for weight in self.parameters():
                weight.data.uniform_(-stdv, stdv)

    def forward(self, g, node_feats, edge_feats):
        ##Layer 1
        h = self.EUGATConv1(g, node_feats, edge_feats) 
        h_0 = th.squeeze(h[0]) ##h_0: node feature, h_1: edge feature
        h_1 = th.squeeze(h[1])
        h_0 = self.embedding_dropout1(h_0)
        h_1 = self.embedding_dropout2(h_1)
        h_0 = F.relu(h_0)+node_feats
        h_1 = F.relu(h_1)+edge_feats
        ##Layer2
        h = self.EUGATConv2(g, h_0, h_1) 
        h_0 = F.relu(h_0)
        h = th.squeeze(h[0])+node_feats
        #h = h[g.ndata["node_mask"].bool()]
        return h
    
class SimpleFFNN(nn.Module):
    def __init__(self, input_dim=768, output_dim=1, dropout_rate=0.1):
        super(SimpleFFNN, self).__init__()
        self.fc = nn.Linear(input_dim, output_dim)  # One fully connected layer
        self.dropout = nn.Dropout(dropout_rate)    # Dropout layer
        self.sigmoid = nn.Sigmoid()                # Sigmoid activation
    
    def forward(self, x):
        x = self.fc(x)             # Fully connected layer
        x = self.dropout(x)        # Apply dropout
        x = self.sigmoid(x)        # Pass through sigmoid
        return x

class CaseGnn(nn.Module):
    def __init__(self, in_dim=768, h_dim=768, out_dim=768, dropout=0.1, num_head=1):
        super(CaseGnn, self).__init__()
        self.hidden_size = h_dim
        self.in_dim = in_dim
        self.eugat_gnn = EUGATGNN(in_dim, h_dim, out_dim, dropout, num_head) 
        self.ffnn = SimpleFFNN()
        
        # self.alpha = nn.Parameter(torch.rand(()))  # Initialize alpha
        # self.beta = nn.Parameter(torch.rand(()))  # Initialize beta

        # self.alpha = nn.Parameter(torch.tensor(0.9))  # Initialize alpha
        # self.beta = nn.Parameter(torch.tensor(0.1))  # Initialize beta

        # self.alpha = 0.9
        # self.beta = 0.1
        
        self.loss = nn.CrossEntropyLoss()
        #self.reset_parameters()

    def reset_parameters(self):
        if self.hidden_size == 0:
            stdv = 1.0 / math.sqrt(self.in_dim)
            for weight in self.parameters():
                weight.data.uniform_(-stdv, stdv)
        else:
            stdv = 1.0 / math.sqrt(self.hidden_size)
            for weight in self.parameters():
                weight.data.uniform_(-stdv, stdv)

    def forward(self,query_pos, candidate_pos, bm25_scores,candidate_relevance_labels, graph):
        
        h = self.eugat_gnn(graph, graph.ndata["h"], graph.edata["h"])
        #QUERY
        query_encoded = h[graph.ndata["query_mask"].bool()][query_pos]
        alphas = self.ffnn(query_encoded)
        #PREC
        h_candidate = h[graph.ndata["candidate_mask"].bool()]
        candidate_encoded = torch.gather(h_candidate.unsqueeze(0).repeat(query_encoded.size(0), 1, 1), 1, candidate_pos.unsqueeze(2).repeat(1, 1, h_candidate.size(1)))
        
        #SCORES CALCULATION
        scores = torch.bmm(query_encoded.unsqueeze(1), candidate_encoded.permute(0,2,1)).squeeze(1)

        mean_scores = scores.mean(dim=1, keepdim=True)
        std_scores = scores.std(dim=1, keepdim=True)
        normalized_scores = (scores - mean_scores) / std_scores
        #scores = self.alpha * normalized_scores + self.beta * bm25_scores
        scores =  alphas * normalized_scores + (1-alphas) * bm25_scores
        #LOSS CALCULATION
        loss = self.loss(scores, candidate_relevance_labels.argmax(dim=1))

        return loss
    
class Test_CaseGnn(CaseGnn):
    def __init__(self,in_dim=768, h_dim=768, out_dim=768, dropout=0.1, num_head=1):
        super().__init__()
        self.eugat_gnn = EUGATGNN(in_dim, h_dim, out_dim, dropout, num_head) 
        self.ffnn = SimpleFFNN()
        # self.alpha = nn.Parameter(torch.rand(()))  # Initialize alpha
        # self.beta = nn.Parameter(torch.rand(()))   # Initialize beta
        # self.alpha = 0.9
        # self.beta = 0.1

    def forward(self,bm25_scores,graph):
        h = self.eugat_gnn(graph, graph.ndata["h"], graph.edata["h"])
        #QUERY
        query_encoded = h[graph.ndata["query_mask"].bool()]
        alphas = self.ffnn(query_encoded)

        #CANDIDATE
        candidate_encoded = h[graph.ndata["candidate_mask"].bool()]
        
        scores = torch.matmul(query_encoded, candidate_encoded.permute(1,0))

        mean_scores = scores.mean(dim=1, keepdim=True)
        std_scores = scores.std(dim=1, keepdim=True)
        normalized_scores = (scores - mean_scores) / std_scores
        #scores = self.alpha * normalized_scores + self.beta * bm25_scores
        scores =  alphas * normalized_scores + (1-alphas) * bm25_scores
        
        #return scores
        return scores,alphas




    

        