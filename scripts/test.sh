#!/usr/bin/env bash

echo "test.sh"
which mpiexec

mpiexec -genvall -n 1 pipenv run test

rm -rf test_tmp
mkdir -p test_tmp
cd test_tmp

head -c 1024 /dev/urandom >test_tmp/random.txt
md5sum test_tmp/*

