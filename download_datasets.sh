#!/bin/bash

wget https://uni-bonn.sciebo.de/s/xzfKyedRRk5DKxy/download
mv download datasets/PlaySlot_Datasets.tar
cd datasets
tar -xvf PlaySlot_Datasets.tar
mv datasets/* .
rm PlaySlot_Datasets.tar
rm -r datasets