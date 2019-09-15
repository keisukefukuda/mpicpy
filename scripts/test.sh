#!/usr/bin/env bash

PROJECT_ROOT=$(dirname $0)/..

if [[ ! -d shunit2 ]]; then
  pushd ${PROJECT_ROOT}/scripts
  git clone https://github.com/kward/shunit2.git
  popd
fi

cd ${PROJECT_ROOT}

DUMMY_FILE_SIZE=32

# Prepare a dummy target file
oneTimeSetUp() {
  rm -rf test_tmp
  mkdir -p test_tmp
  F="test_tmp/random.txt"
  head -c ${DUMMY_FILE_SIZE} /dev/urandom >${F}
  SUM_ORIG=$(md5sum "${F}" | cut -f 1 -d ' ')
}

setUp() {
  echo "------------------------- setUp() ------------------------"
  echo rm -f ${F}.*  # Remove all files except the original
  rm -f ${F}.*  # Remove all files except the original
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

  ### If the file $F.1 exists, next call must be an error
  ls test_tmp/
  mpiexec -n 2 python mpicpy/mpicpy.py "${F}.{rank}" --rank=0 2>/dev/null
  assertEquals 2 $?

  ### rank 1 --> rank 0 with -f
  # Generate a new random ${F}.1
  head -c 1024 /dev/urandom >${F}.1
  SUM1=$(md5sum "${F}.1" | cut -f 1 -d ' ')

  # Copy rank 1 --> rank 0
  mpiexec -n 2 python mpicpy/mpicpy.py "${F}.{rank}" --rank=1 -f 2>/dev/null
  echo "(3)"
  md5sum test_tmp/*
  SUM0=$(md5sum ${F}.0 | cut -f 1 -d ' ')
  assertEquals "${SUM1}" "${SUM0}"
}

testSmallChunk() {
  cp "${F}" "${F}.0"

  for CHUNK_SIZE in 1 2 3 4 5; do
      rm -f ${F}.1
      SUM0=$(md5sum ${F}.0 | cut -f 1 -d ' ')
      mpiexec -n 2 python mpicpy/mpicpy.py "${F}.{rank}" --rank=0 -c $CHUNK_SIZE
      assertEquals 0 $?
      SUM1=$(md5sum "${F}.1" | cut -f 1 -d ' ')
      assertEquals "${SUM0}" "${SUM1}"
  done
}

. ${PROJECT_ROOT}/scripts/shunit2/shunit2

