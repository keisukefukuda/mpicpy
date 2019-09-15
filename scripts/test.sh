#!/usr/bin/env bash

PROJECT_ROOT=$(dirname $0)/..

if [[ ! -d shunit2 ]]; then
  pushd ${PROJECT_ROOT}/scripts
  git clone https://github.com/kward/shunit2.git
  popd
fi

cd ${PROJECT_ROOT}

# Prepare a dummy target file
oneTimeSetUp() {
  rm -rf test_tmp
  mkdir -p test_tmp
  F="test_tmp/random.txt"
  head -c 1024 /dev/urandom >${F}
  SUM_ORIG=$(md5sum "${F}" | cut -f 1 -d ' ')
}

setUp() {
  echo "------------------------- setUp() ------------------------"
  echo rm -f ${F}.*  # Remove all files except the original
  rm -f ${F}.*  # Remove all files except the original
}

tearDown() {
  echo
}

testSize()
{
  cp "${F}" "${F}.0"

  # Copy ${F}.0 --> ${F}.1
  mpiexec -n 2 python mpicpy/mpicpy.py "${F}.{rank}" --size
  local SUM1=$(md5sum "${F}.1" | cut -f 1 -d ' ')
  assertEquals ${SUM_ORIG} ${SUM1}

  # If the file $F.1 exists, next call must be an error
  mpiexec -n 2 python mpicpy/mpicpy.py "${F}.{rank}" --size 2>/dev/null
  assertEquals 2 $?

  # If $F is empty (i.e. there is $F.0 and $F.1)
  # $F.1 is copied to ${F}.0 with -f option.
  echo >${F}.0
  mpiexec -n 2 python mpicpy/mpicpy.py "${F}.{rank}" -f --size
  local SUM0=$(md5sum ${F}.0 | cut -f 1 -d ' ')
  local SUM1=$(md5sum ${F}.1 | cut -f 1 -d ' ')
  assertEquals "${SUM_ORIG}" "${SUM0}"
  assertEquals "${SUM_ORIG}" "${SUM1}"
}

testMD5() {
  cp "${F}" "${F}.0"

  SUM0=$(md5sum ${F}.0 | cut -f 1 -d ' ')
  mpiexec -n 2 python mpicpy/mpicpy.py "${F}.{rank}" --md5="${SUM0}"
  assertEquals 0 $?
  SUM1=$(md5sum "${F}.1" | cut -f 1 -d ' ')
  assertEquals "${SUM0}" "${SUM1}"

  # If the file $F.1 exists, next call must be an error
  mpiexec -n 2 python mpicpy/mpicpy.py "${F}.{rank}" --md5="${SUM0}" 2>/dev/null
  assertEquals 2 $?

  # Copy ${F}.0 --> ${F}.1
  # what if ${F}.0 and ${F}.1 are different?
  head -c 1024 /dev/urandom >${F}.1
  # It must be an error without -f
  mpiexec -n 2 python mpicpy/mpicpy.py "${F}.{rank}" --md5="${SUM0}" 2>/dev/null
  assertEquals 2 $?

  # now with -f
  mpiexec -n 2 python mpicpy/mpicpy.py "${F}.{rank}" --md5="${SUM0}" -f 2>/dev/null
  SUM1=$(md5sum "${F}.1" | cut -f 1 -d ' ')
  assertEquals "${SUM0}" "${SUM1}"
}

testRank() {
  cp "${F}" "${F}.0"

  SUM0=$(md5sum ${F}.0 | cut -f 1 -d ' ')
  mpiexec -n 2 python mpicpy/mpicpy.py "${F}.{rank}" --rank=0
  assertEquals 0 $?
  SUM1=$(md5sum "${F}.1" | cut -f 1 -d ' ')
  assertEquals "${SUM0}" "${SUM1}"

  # If the file $F.1 exists, next call must be an error
  ls test_tmp/
  mpiexec -n 2 python mpicpy/mpicpy.py "${F}.{rank}" --rank=0 2>/dev/null
  assertEquals 2 $?

  # rank 1 --> rank 0 with -f
  #head -c 1024 /dev/urandom >${F}.1
  #mpiexec -n 2 python mpicpy/mpicpy.py "${F}.{rank}" --rank=1 -f 2>/dev/null
  #SUM0=$(md5sum ${F}.0 | cut -f 1 -d ' ')
  #assertEquals "${SUM0}" "${SUM1}"
}

. ${PROJECT_ROOT}/scripts/shunit2/shunit2

