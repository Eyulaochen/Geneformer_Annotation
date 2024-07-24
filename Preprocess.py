import pickle
import scanpy as sc
import numpy as np
from collections import Counter
from datasets import load_from_disk
from geneformer.pretrainer import token_dictionary
with open('gene_name_id_dict.pkl', 'rb') as f:
    names = pickle.load(f)
with open('gene_median_dictionary.pkl', 'rb') as f:
    median = pickle.load(f)

data = sc.read_h5ad('full/VAL_snRNA2_9M.h5ad', backed='r')
X = data.X[:]
X = X.toarray()
var = data.var.index

