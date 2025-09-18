#!/usr/bin/env python
# coding: utf-8

# In[ ]:


rs =metrics_at_k_all(test_gold_Scores, prec_score)

print(rs["map"])
print(rs["mrr"])
print(max(rs["mF"]))
print(alphas.mean())

