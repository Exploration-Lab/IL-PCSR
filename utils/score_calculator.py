
import torch
from tqdm import tqdm

from itertools import product
import random
import pickle
from utils.dataset import get_metadata,get_gold

metadata = get_metadata()
gold = get_gold()

def get_colieee_gold_matrix(query_index2id,candidate_index2id,candidate_type,gold):
    gold_matrix = torch.empty(len(query_index2id), len(candidate_index2id))

    for i,query_id in enumerate(query_index2id):

        if query_id in gold:
            
            expected = gold[query_id]
        else:
            expected = []
            print("Missing key: ", query_id)

        for j,candidate_sec_id in enumerate(candidate_index2id):
            
            if candidate_sec_id in expected:
                gold_matrix[i][j] = 1
            else:
                gold_matrix[i][j] = 0 

    #print(gold_matrix.shape)
    return gold_matrix

def get_gold_matrix(query_index2id,candidate_index2id,candidate_type):
    gold_matrix = torch.empty(len(query_index2id), len(candidate_index2id))

    for i,query_id in enumerate(query_index2id):

        expected = gold[query_id][candidate_type]

        for j,candidate_sec_id in enumerate(candidate_index2id):
            
            if candidate_sec_id in expected:
                gold_matrix[i][j] = 1
            else:
                gold_matrix[i][j] = 0 

    print(gold_matrix.shape)
    return gold_matrix


def get_ranking_score(model,graph,gold_matrix,query_index2id,candidate_index2id,k=11):
    with torch.no_grad():
        preds = model(graph)

    test_score = preds.view(len(query_index2id), len(candidate_index2id))

    return metrics_at_k(gold_matrix, test_score,top_k=k)


def metrics_at_k(G, S, top_k=11):
    num_samples, num_items = G.shape

    prec = torch.zeros((num_samples, top_k-1))
    rec = torch.zeros((num_samples, top_k-1))
    f1 = torch.zeros((num_samples, top_k-1))
    mrr_min = torch.zeros((num_samples,))

    for i in range(num_samples):
        s = S[i]
        g = G[i]
        
        s, x = torch.sort(s, descending=True)
        g = g[x]
        
        nz = g.nonzero(as_tuple=True)[0]
        
        if len(nz) > 0:
            mrr_min[i] = 1/(nz[0]+1)
        else:
            mrr_min[i] = 0        
            
        t=0
        for k in range(1,top_k,1):
            _g = g[:k]
            P = _g.sum()/k
            R = _g.sum()/g.sum() if g.sum() > 0 else 0
            F = 2 * P * R / (P + R) if P + R != 0 else 0
            prec[i, t] = P
            rec[i, t] = R
            f1[i, t] = F
            t += 1

    PP = prec.mean(axis=0).tolist()
    RR = rec.mean(axis=0).tolist()
    FF = f1.mean(axis=0).tolist()
    MRR_min = mrr_min.mean().item()
    return PP, RR, FF, MRR_min

def metrics_at_k_all(G, S, bS=None):
    # print(G.shape)
    num_samples, num_items = G.shape

    prec = torch.zeros((num_samples, 10))
    rec = torch.zeros((num_samples, 10))
    f1 = torch.zeros((num_samples, 10))
    mrr = []
    map = []
    correct = torch.zeros((num_samples, 10))
    total = torch.zeros((num_samples, 10))

    for i in range(num_samples):
    #for i in tqdm(range(num_samples)):
        s = S[i]
        g = G[i]
        
        if bS is not None: 
            b = bS[i]
            b, y = torch.sort(b, descending=True)
            remove = y[int(len(y)/10):]
            s[remove] = -torch.inf
        
        s, x = torch.sort(s, descending=True)
        g = g[x]
        
        nz = g.nonzero(as_tuple=True)[0].tolist()
        
        if len(nz) > 0:
            mrr.append(1/(nz[0]+1))
        else:
            mrr.append(0)

        if len(nz) > 0:
            map.append(sum((i+1)/(k+1) for i,k in enumerate(nz))/len(nz))
        else:
            map.append(0)
            
        
        for k in list(range(1,11,1)):
            # print("k =", k)
            _g = g[:k]
            P = _g.sum()/k
            R = _g.sum()/g.sum() if g.sum() > 0 else 0
            F = 2 * P * R / (P + R) if P + R != 0 else 0
            prec[i, k-1] = P
            rec[i, k-1] = R
            f1[i, k-1] = F
            correct[i, k-1] = _g.sum()
            total[i, k-1] = g.sum()
            
    mrr = sum(mrr) / num_samples
    map = sum(map) / num_samples

    mP = prec.mean(axis=0)
    mR = rec.mean(axis=0)
    mF = f1.mean(axis=0)
    
    muP = correct.sum(axis=0) / (torch.arange(1, 11) * num_samples)
    muR = correct.sum(axis=0) / total.sum(axis=0)
    muF = 2 * muP * muR / (muP + muR)

    return {'mP': mP.tolist(), 'mR': mR.tolist(), 'mF': mF.tolist(), 'muP': muP.tolist(), 'muR': muR.tolist(), 'muF': muF.tolist(), 'mrr': mrr, 'map': map}


def metrics_at_k_individual(G, S, top_k=11):
    num_samples, num_items = G.shape

    prec = torch.zeros((num_samples, top_k-1))
    rec = torch.zeros((num_samples, top_k-1))
    f1 = torch.zeros((num_samples, top_k-1))
    mrr_min = torch.zeros((num_samples,))

    for i in range(num_samples):
        s = S[i]
        g = G[i]
        
        s, x = torch.sort(s, descending=True)
        g = g[x]
        
        nz = g.nonzero(as_tuple=True)[0]
        
        if len(nz) > 0:
            mrr_min[i] = 1/(nz[0]+1)
        else:
            mrr_min[i] = 0        
            
        t=0
        for k in range(1,top_k,1):
            _g = g[:k]
            P = _g.sum()/k
            R = _g.sum()/g.sum() if g.sum() > 0 else 0
            F = 2 * P * R / (P + R) if P + R != 0 else 0
            prec[i, t] = P
            rec[i, t] = R
            f1[i, t] = F
            t += 1

    # PP = prec.mean(axis=0).tolist()
    # RR = rec.mean(axis=0).tolist()
    # FF = f1.mean(axis=0).tolist()
    # MRR_min = mrr_min.mean().item()
    return prec, rec, f1, mrr_min
    # return PP, RR, FF, MRR_min