import sys
import numpy as np
import scanpy as sc
from collections import Counter

train = sc.read_h5ad(sys.argv[1], backed='r')
#print(Counter(train.obs['celltype']))

classes_to_downsample = []
downsample = int(sys.argv[2])
for celltype in train.obs['celltype'].unique():
    cell_indices = np.where(train.obs['celltype'] == celltype)[0]
    if (len(cell_indices)>downsample):
        classes_to_downsample.append([celltype, len(cell_indices)])

#print(classes_to_downsample)
#print(len(classes_to_downsample))

classes_to_downsample = [x[0] for x in classes_to_downsample]

selected_indices = []
for class_label in classes_to_downsample:
    class_indices = np.where(train.obs['celltype'] == class_label)[0]
    factor = downsample/len(class_indices)
    selected_class_indices = []
    for sample in list(Counter(train.obs['sampleid'])):
        class_sample = np.where((train.obs['celltype'] == class_label) & (train.obs['sampleid'] == sample))[0]
        num_samples_to_del = int(len(class_sample) * (1-factor))
        selected_class_sample = np.random.choice(class_sample, size=num_samples_to_del, replace=False)
        selected_class_indices.append(selected_class_sample)
    selected_class_indices_ = np.concatenate(selected_class_indices)
    selected_indices.append(selected_class_indices_)
    
all_selected_indices = np.concatenate(selected_indices)
remaining_indices = np.setdiff1d(np.arange(len(train)), all_selected_indices)
downsampled_adata = train[remaining_indices]

name = sys.argv[1].split('/')[-1]
downsampled_adata.write(sys.argv[1][:-len(name)]+'DownSampled_'+name, compression='gzip', compression_opts=3)
