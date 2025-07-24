#!/bin/bash

# check the directory for master.zip
# if master.zip does not exist, fetch it via download

test ! -f "master.zip" && wget "https://github.com/karoldvl/ESC-50/archive/master.zip"

# unzip master.zip

unzip -qq master.zip

# set up needed directories for additional test data and model storage

mkdir data && mkdir models

# change directory to data and fetch additional test data via download

cd data

wget 'http://soundbible.com/grab.php?id=2215&type=wav' -O "dog.wav"
wget "http://soundbible.com/grab.php?id=1954&type=wav" -O "cat.wav"

cd ../