import torch
def get_rank_list(q_ids,c_ids,scores):
    rank_list = {}
    for i, qid in enumerate(q_ids):
        score = scores[i]
        _,indices  = torch.sort(score,descending=True)
        ids = [c_ids[j] for j in indices]
        rank_list[qid] = ids
    return rank_list
    