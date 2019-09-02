import argparse
import os.path
import re

import fire
from mpi4py import MPI

comm = MPI.COMM_WORLD


def decide_root(filepath, root):
    m = re.match(r'^\d+$', root)
    if m:
        root = int(m)
    elif root == 'auto':
        # root would be the first process that owns the file
        exists = comm.allgather(os.path.exists(filepath))

        # get the fisrt index that 'exists' is True
        ranks = [idx for idx, val in enumerate(exists) if val is True]

        if len(ranks) == 0:
            raise RuntimeError("No rank has the file '{}'".format(filepath))

        root = ranks[0]
    else:
        raise ValueError("Invalid value for 'root' option.")

    if root == comm.rank:
        if not os.path.exists(filepath):
            raise RuntimeError('Bcast root {} does not have the file "{}"'.format(
                root, filepath
            ))

        if not os.path.isfile(filepath):
            raise RuntimeError('"{}" is not a regular file.'.format(
                filepath
            ))

    return root


def parse_chunksize(s):
    if type(s) == int:
        assert s >= 0
        return s

    m = re.match(r'^(\d+)([kmg]?)$', s, re.I)

    if m is None:
        raise RuntimeError('Cannot parse chunksize: "{}"'.format(s))

    digits = int(m.group(1))

    suffix = m.group(2).lower()

    if suffix == 'k':
        digits *= 1024
    elif suffix == 'm':
        digits *= 1024 * 1024
    elif suffix == 'g':
        digits == 1024 * 1024 * 1024
    else:
        assert False

    return digits


def send_file(filepath, chunksize):
    size = os.path.getsize(filepath)

    assert size >= 0

    print("Rank {} [Root] size={}".format(comm.rank, size))
    comm.bcast(size, root=comm.rank)

    if size == 0:
        num_chunks = 0
    else:
        num_chunks = ((size - 1) // chunksize) + 1

    print("num_chunks = {}".format(num_chunks))

    with open(filepath, 'rb') as f:
        for i in range(num_chunks):
            print("Sending Chunk #{}".format(i+1))
            buf = f.read(chunksize)
            comm.Bcast(buf, root=comm.rank)


def recv_file(root, filepath, chunksize):
    size = None
    size = comm.bcast(size, root=root)
    print("Rank {} size={}".format(comm.rank, size))

    if size == 0:
        num_chunks = 0
    else:
        num_chunks = ((size - 1) // chunksize) + 1

    print("\t\tnum_chunks = {}".format(num_chunks))

    with open(filepath, 'wb') as f:
        for i in range(num_chunks):
            if i < num_chunks - 1:
                buf = bytearray(chunksize)
            else:
                buf = bytearray(size % chunksize)

            print("\t\tReceiving Chunk #{} from {}".format(i+1, root))
            comm.Bcast(buf, root=root)
            f.write(buf)


def main(filepath, root='auto', chunksize='4m', rank_suffix=False):
    root = decide_root(filepath, root)
    assert type(root) == int
    assert 0 <= root and root < comm.size

    if rank_suffix:
        local_filename = '{}.{}'.format(filepath, comm.rank)

    chunksize = parse_chunksize(chunksize)

    if comm.rank == root:
        send_file(filepath, chunksize)
    else:
        recv_file(root, local_filename, chunksize)


if __name__ == '__main__':
    fire.Fire(main)