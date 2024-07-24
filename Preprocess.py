import pickle
import scanpy as sc
import numpy as np
from collections import Counter
from datasets import Dataset
from datasets import load_from_disk
from geneformer.pretrainer import token_dictionary
with open('gene_name_id_dict.pkl', 'rb') as f:
    names = pickle.load(f)
with open('gene_median_dictionary.pkl', 'rb') as f:
    median = pickle.load(f)

def process(data_dir):
    data = sc.read_h5ad(data_dir, backed='r')
    X = data.X[:]
    X = X.toarray()
    var = data.var.index

    l = []
    indvec = []
    ct = 0
    for v in var:
        if (v not in names.keys()) or (names[v] not in token_dictionary.keys()):
            l.append(ct)
            ct = ct+1
        else:
            indvec.append(token_dictionary[names[v]])
            X[:,ct] /=(median[names[v]])
            ct = ct+1
    X = np.delete(X, l, 1)
    X.shape
    
    data_list = []
    for i in range(len(X)):
        argsort = np.argsort(X[i,:])
        c = len(argsort) - np.count_nonzero(X[i,:])
        data_list.append(np.array(indvec)[argsort[c:].astype(int)])

    # cut > 2048
    for i in range(len(data_list)):
        # reverse order
        data_list[i] = np.flip(data_list[i])
        if (len(data_list[i]) > 2048):
            data_list[i] = data_list[i][:2048]

    data_list_ = []
    for i in range(len(data_list)):
        data_list_.append(data_list[i].copy().tolist())

    lengths = []
    for i in range(len(data_list_)):
        lengths.append(len(data_list_[i]))
    
    data_dict = {"label": data.obs['celltype'], "input_ids": data_list_, "length": lengths}
    ds = Dataset.from_dict(data_dict)
    ds.save_to_disk(data_dir[:-5] + '.dataset')

data_dir = 'data/VAL_snRNA2_9M.h5ad'
process(data_dir)

test = load_from_disk(data_dir[:-5] + '.dataset')
print(test[0])
