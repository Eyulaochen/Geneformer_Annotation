#! /bin/bash

name='HRCAv2_snRNA'
mkdir data/$name

python Training_Val_Divide.py "data/${name}.h5ad"
python DownSampling.py "data/${name}/TRAIN_${name}.h5ad" 5000

python Preprocess.py "data/${name}/DownSampled_TRAIN_${name}.h5ad"
python Preprocess.py "data/${name}/VAL_${name}.h5ad"

python Numerical_Labels.py "data/${name}/DownSampled_TRAIN_${name}.h5ad"
python Fine_Tuning.py "2_9M_111celltype.npy" "data/${name}/DownSampled_TRAIN_${name}.dataset" "data/${name}/VAL_${name}.dataset" "output_models"


#touch data/${name}_results
#python Evaluation.py "checkpoint-1428300" "data/${name}_results"

