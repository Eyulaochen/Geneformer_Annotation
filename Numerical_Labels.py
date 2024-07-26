

# prepare the numerical labels
training = sc.read_h5ad('data/downsampled372K_snrna.h5ad', backed='r')
target_names = list(Counter(training.obs['celltype']))
target_name_id_dict = dict(zip(target_names,[i for i in range(len(target_names))]))
np.save('2_9M_111celltype.npy', target_name_id_dict)
#target_name_id_dict = np.load('2_9M_111celltype.npy',allow_pickle='TRUE').item()
del training
