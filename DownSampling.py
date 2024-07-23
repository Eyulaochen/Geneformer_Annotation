import scanpy as sc
from collections import Counter

train29 = sc.read_h5ad('full/TRAIN_snRNA2_9M.h5ad', backed='r')
print(Counter(train29.obs['celltype']))

classes_to_downsample = []
downsample = 5000
for celltype in train29.obs['celltype'].unique():
    cell_indices = np.where(train29.obs['celltype'] == celltype)[0]
    if (len(cell_indices)>downsample):
        classes_to_downsample.append([celltype, len(cell_indices)])

print(classes_to_downsample)
print(len(classes_to_downsample))

classes_to_downsample = [x[0] for x in classes_to_downsample]
