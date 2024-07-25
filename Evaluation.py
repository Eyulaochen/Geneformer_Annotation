import os
import scanpy as sc
import torch
import pickle
import numpy as np
from sklearn.metrics import f1_score
from sklearn.metrics import accuracy_score
from collections import Counter
from datasets import load_from_disk
from datasets import Dataset
from geneformer.pretrainer import token_dictionary
with open('gene_name_id_dict.pkl', 'rb') as f:
    names = pickle.load(f)
with open('gene_median_dictionary.pkl', 'rb') as f:
    median = pickle.load(f)

target_name_id_dict = np.load('2_9M_111celltype.npy',allow_pickle='TRUE').item()

