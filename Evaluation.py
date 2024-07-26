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
from Preprocess import process
with open('gene_name_id_dict.pkl', 'rb') as f:
    names = pickle.load(f)
with open('gene_median_dictionary.pkl', 'rb') as f:
    median = pickle.load(f)

target_name_id_dict = np.load('2_9M_111celltype.npy',allow_pickle='TRUE').item()

def classes_to_ids(example):
    example["label"] = target_name_id_dict[example["label"]]
    return example

model = BertForSequenceClassification.from_pretrained("checkpoint-1428300", 
                                                      num_labels=len(target_name_id_dict),
                                                      output_attentions = False,
                                                      output_hidden_states = False).to(cuda)

def evaluation(val_dir, fmt = 'h5ad', cuda = 'cuda:0'):
    if (fmt == 'h5ad'):
        process(val_dir)
        val = load_from_disk(val_dir[:-5] + '.dataset')
    elif(fmt == 'dataset'):
        val = load_from_disk(val_dir[:-5] + '.dataset')
    evalset = val.map(classes_to_ids, num_proc=16)
    List = torch.tensor(())
    for num in range(len(evalset)):
        x = torch.tensor(evalset[num]['input_ids']).to(cuda)
        y = model(x.unsqueeze(0)).logits
        List = torch.cat((List, y[0].argmax().to('cpu').unsqueeze(0)), 0)
    torch.save(List, val_dir[:-5] + '.pth')

def PR_f1(val_dir, val_label):
    temp = val_dir.split('/')
    val = sc.read_h5ad(val_dir, backed='r')
    results = torch.load(val_label)
    file = open(val_dir[:-len(temp[-1])-1] + '_results/' +  temp[-1][:-5] + '.txt', 'w')
    pre = 0
    rec = 0
    f1 = 0
    for celltype in target_name_id_dict.keys():
        temp = (val.obs['celltype'] == celltype).tolist()
        indices = [i for i in range(len(temp)) if temp[i] == True]
        if (len(indices) == 0):
            print(celltype)
        result = (results == target_name_id_dict[celltype]).nonzero(as_tuple=True)[0]
        if ((len(indices) != 0) and (len(result) == 0)):
            file.write(celltype + ' ' + str(len(indices)) + ':' + '0.0' + ' ' '\n')
        if ((len(indices) != 0) and (len(result) != 0)):
            inter = len(set(result.tolist()).intersection(indices))
            precision = inter/len(result)
            recall = inter/len(indices)
            pre = pre + precision
            rec = rec + recall
            f1 = f1 + 2*precision*recall/(precision+recall)
            file.write(celltype + ' ' + str(len(indices)) + ':' + str(precision) + ' ' 
                       + str(recall) + '\n')
    label_list = [target_name_id_dict[celltype] for celltype in val.obs['celltype']]
    file.write('\n' + '\n' + '\n')
    file.write('acurracy:' + str(accuracy_score(label_list, results.tolist())) + '\n')
    file.write('ave-pre:' + str(pre/len(Counter(val.obs['celltype']))) + '\n')
    file.write('ave-rec:' + str(rec/len(Counter(val.obs['celltype']))) + '\n')
    file.write('macro_f1:' + str(f1/len(list(Counter(val.obs['celltype'])))) + '\n')

