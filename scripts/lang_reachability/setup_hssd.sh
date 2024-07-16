#!/bin/bash

data_source="https://huggingface.co/datasets/hssd/hssd-scenes/resolve/main/scenes/"
scenes_names=("102344022" "102344094" "102344469" "102815859" "102816216" "103997403_171030405")

mkdir -p data
mkdir -p data/hssd
cd data/hssd

for t in "${scenes_names[@]}"; do
    if [ ! -f $t".glb" ]; then
        wget -O $t".glb" $data_source"$t".glb
    fi
done

cd ../..


# https://huggingface.co/datasets/hssd/hssd-scenes/resolve/main/scenes/102816114.glb?download=true
