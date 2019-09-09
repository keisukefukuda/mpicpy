#!/usr/bin/env bash

echo "test.sh"
which mpiexec

mpiexec -genvall -n 1 pipenv run test

