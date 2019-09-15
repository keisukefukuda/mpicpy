#!/usr/bin/env bash

PROJECT_ROOT=$(dirname $0)/..

if [[ ! -d shunit2 ]]; then
  pushd ${PROJECT_ROOT}/scripts
  git clone https://github.com/kward/shunit2.git
  popd
fi

cd ${PROJECT_ROOT}

# Prepare a dummy target file
rm -rf test_tmp
mkdir -p test_tmp
F="test_tmp/random.txt"
head -c 1024 /dev/urandom >${F}
cp "${F}" "${F}.orig"

SUM_ORIG=$(md5sum "${F}.orig" | cut -f 1 -d ' ')

testSize()
{
  cp "${F}" "${F}.0"

  # Copy ${F}.0 --> ${F}.1
  echo "#### 1"
  mpiexec -n 2 python mpicpy/mpicpy.py "${F}.{rank}" --size
  SUM1=$(md5sum "${F}.1" | cut -f 1 -d ' ')
  assertEquals ${SUM_ORIG} ${SUM1}

  # If the file $F.1 exists, next call must be an error
  echo "#### 2"
  mpiexec -n 2 python mpicpy/mpicpy.py "${F}.{rank}" --size 2>/dev/null
  assertEquals 2 $?

  # If $F is empty (i.e. there is $F.0 and $F.1)
  # $F.1 is copied to ${F}.0 with -f option.
  echo "#### 3"
  echo >${F}.0
  echo "mpiexec -n 2 python mpicpy/mpicpy.py ${F}.{rank} -f --size"
  mpiexec -n 2 python mpicpy/mpicpy.py "${F}.{rank}" -f --size
  SUM0=$(md5sum ${F}.0 | cut -f 1 -d ' ')
  SUM1=$(md5sum ${F}.1 | cut -f 1 -d ' ')
  assertEquals "${SUM_ORIG}" "${SUM0}"
  assertEquals "${SUM_ORIG}" "${SUM1}"
}


. ${PROJECT_ROOT}/scripts/shunit2/shunit2

