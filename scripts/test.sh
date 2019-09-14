#!/usr/bin/env bash

echo "test.sh"
which mpiexec

mpiexec -genvall -n 1 pipenv run test

rm -rf test_tmp
mkdir -p test_tmp

F=test_tmp/random.txt
head -c 1024 /dev/urandom >$F
SUM=$(md5sum $F | cut -f 1 -d ' ')


mpiexec -n 2 python mpicpy/mpicpy.py $F --rank-suffix --size
SUM2=$(md5sum $F.1 | cut -f 1 -d ' ')

echo SUM=$SUM
echo SUM2=$SUM2

