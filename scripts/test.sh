#!/usr/bin/env bash

PROJECT_ROOT=$(dirname $0)/..

if [ ! -d shunit2 ]; then
  pushd ${PROJECT_ROOT}/scripts
  git clone https://github.com/kward/shunit2.git
  popd
fi

cd $PROJECT_ROOT
mpiexec -genvall -n 1 pipenv run test

rm -rf test_tmp
mkdir -p test_tmp
F=test_tmp/random.txt

head -c 1024 /dev/urandom >$F
SUM=$(md5sum $F | cut -f 1 -d ' ')

testSize()
{
  mpiexec -n 2 python mpicpy/mpicpy.py $F --rank-suffix --size
  SUM2=$(md5sum $F.1 | cut -f 1 -d ' ')

  assertEquals $SUM $SUM2
}

. ${PROJECT_ROOT}/scripts/shunit2/shunit2

