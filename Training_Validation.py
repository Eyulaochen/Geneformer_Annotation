import scanpy as sc
from collections import Counter

full29 = sc.read_h5ad('full/HRCAv2_snRNA.h5ad', backed='r')
print(Counter(full29.obs['celltype']))
