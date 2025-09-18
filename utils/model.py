
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
from utils.eugatgnn import EUGATGNN
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