#!/bin/bash

### Feature Table ###
# ca_housing 8
# YearPredictionMSD 90
# slice_localization 384
dataset=dataset

python -u ./../src/grid_computing_forecasting/modeling/regression/main_reg_cv.py \
    --feat_d 355 \
    --hidden_d 32 \
    --boost_rate 1 \
    --lr 0.005 \
    --L2 .0e-3 \
    --num_nets 20 \
    --data ${dataset} \
    --tr ${ROOT_DIR}/data/interim/${dataset}_tr.npz \
    --te ${ROOT_DIR}/data/interim/${dataset}_te.npz \
    --batch_size 2048 \
    --epochs_per_stage 1 \
    --correct_epoch 1 \
    --normalization True \
    --cv True \
    --out_f ${ROOT_DIR}/models/${dataset}_cls.pth \
