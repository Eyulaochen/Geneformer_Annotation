import sys
import scanpy as sc
import numpy as np
from collections import Counter

# prepare the numerical labels
training = sc.read_h5ad(sys.argv[1], backed='r')
target_names = list(Counter(training.obs['celltype']))
target_name_id_dict = dict(zip(target_names,[i for i in range(len(target_names))]))
np.save('2_9M_111celltype.npy', target_name_id_dict)
#target_name_id_dict = np.load('2_9M_111celltype.npy',allow_pickle='TRUE').item()
#del training
