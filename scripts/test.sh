#!/usr/bin/env bash

echo "test.sh"
which mpiexec

mpiexec -genvall -n 4 pipenv run test

