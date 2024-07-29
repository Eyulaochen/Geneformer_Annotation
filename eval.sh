#! /bin/bash

name='HRCAv2_snRNA'
mkdir data/${name}_results
python Evaluation.py "checkpoint-1428300" "data/${name}/VAL_${name}.dataset"
