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
