import scanpy as sc
from collections import Counter

full29 = sc.read_h5ad('full/HRCAv2_snRNA.h5ad', backed='r')
print(Counter(full29.obs['celltype']))

data_dict = {
        'Chen_a_10x3_Lobe_19_D003_Nu': 'no_enriched',
        'Chen_a_10x_Lobe_D28_13_NeuN': 'bc_enriched',
        'BCM_22_0458_RNA_macular_NeuN': 'ac_enriched',
        'BCM_22_0784_RNA_macular_NeuN': 'rgc_enriched',
        'Chen_b_D001-12_lobe_NeuNM': 'amd_sample1',
        'Chen_b_D001-12_lobe_NeuNT': 'amd_sample2'
            }

train29 = full29[~full29.obs['sampleid'].isin(data_dict.keys())]
val29 = full29[full29.obs['sampleid'].isin(data_dict.keys())]

train29.write('full/TRAIN_snRNA2_9M.h5ad', compression='gzip', compression_opts=3)
val29.write('full/VAL_snRNA2_9M.h5ad', compression='gzip', compression_opts=3)
