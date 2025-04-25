#!/bin/bash

#download conda install file
wget https://github.com/conda-forge/miniforge/releases/latest/download/Miniforge3-Linux-x86_64.sh

#make install file executable
chmod u+x Miniforge3-Linux-x86_64.sh

#install conda on work partition
./Miniforge3-Linux-x86_64.sh -p ${WORK}/miniforge3

#replace UUSERNAME in initConda.sh with your username
sed -i "s/USERNAME/$USER/" initConda.sh

#make initConda.sh executable
chmod u+x initConda.sh

#move to root directory
mv initConda.sh ~

#delete conda install file
rm Miniforge3-Linux-x86_64.sh

