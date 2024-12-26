#! /bin/bash
conda install -c conda-forge gcc=11.4.0 -y
conda install -c conda-forge gxx=11.4.0 -y

# run C++ in Jupyter Notebook
# https://chatgpt.com/share/676d23f6-d180-8005-b4ce-c8a5d9148fa0
conda install -c conda-forge xeus-cling
