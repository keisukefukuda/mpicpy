#!/bin/sh

set -e

# For TravisCI environment, install custom mpich locally.

if [[ ! -f $HOME/mpich/bin/mpicxx ]]; then
    cd /tmp
    wget http://www.mpich.org/static/downloads/3.2/mpich-3.2.tar.gz 
    tar -zxf /tmp/mpich-3.2.tar.gz
    cd mpich-3.2
    ./configure --prefix=$HOME/mpich --disable-fortran && make && make install
    
    rm -rf /tmp/mpich-*
else
    echo "MPICH is already installed."
fi
