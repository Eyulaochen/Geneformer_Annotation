import sys 
import scanpy as sc
from collections import Counter

full = sc.read_h5ad(sys.argv[1], backed='r')
name = sys.argv[1].split('/')[-1]
#print(Counter(full.obs['celltype']))

data_dict = {
        'Chen_a_10x3_Lobe_19_D003_Nu': 'no_enriched',
        'Chen_a_10x_Lobe_D28_13_NeuN': 'bc_enriched',
        'BCM_22_0458_RNA_macular_NeuN': 'ac_enriched',
        'BCM_22_0784_RNA_macular_NeuN': 'rgc_enriched',
        'Chen_b_D001-12_lobe_NeuNM': 'amd_sample1',
        'Chen_b_D001-12_lobe_NeuNT': 'amd_sample2'
            }

train = full[~full.obs['sampleid'].isin(data_dict.keys())]
val = full[full.obs['sampleid'].isin(data_dict.keys())]

train.write(sys.argv[1][:-5] + '/TRAIN_' + name, compression='gzip', compression_opts=3)
val.write(sys.argv[1][:-5] + '/VAL_' + name, compression='gzip', compression_opts=3)
