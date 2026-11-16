#! /bin/bash
set -e

conda install -c conda-forge gcc=15.3.0 gxx=15.3.0 -y
# conda install -c conda-forge cxx-compiler -y

# run C++ in Jupyter Notebook
# conda install -c conda-forge xeus-cpp=0.9.0 -y
conda install -c conda-forge gdb ncurses -y
