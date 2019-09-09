#!/usr/bin/env bash

mpiexec -x PATH,LD_LIBRARY_PATH -n 4 pipenv run test
