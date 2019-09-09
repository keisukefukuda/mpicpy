#!/usr/bin/env bash

echo "test.sh"
which mpiexec

mpiexec -x PATH,LD_LIBRARY_PATH -n 4 pipenv run test

