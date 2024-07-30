#! /bin/bash

name='HRCAv2_snRNA'
mkdir data/${name}_results
python Evaluation.py "2_9M_111celltype.npy"  "output_models/checkpoint-1428300" "data/${name}/VAL_${name}.h5ad"
