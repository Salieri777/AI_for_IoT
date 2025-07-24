#!/bin/bash

# check to see if ESC-50-master directory exist

if [ -d '/ESC-50-master' ];

then
    # if it does - train both models and generate comparative predictions
    python train.py
    python predict.py

else
    # if it does not - fetch the necessary datat
    # train both models and generate comparative predictions
    bash fetch_data.sh
    python train.py
    python predict.py

fi