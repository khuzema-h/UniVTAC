#!/bin/bash

# Download the dataset using modelscope
if ! command -v modelscope &> /dev/null
then
    pip install modelscope
fi

# modelscope download --dataset byml2024/UniVTAC  
modelscope download --dataset byml2024/UniVTAC --include "insert_HDMI/**"