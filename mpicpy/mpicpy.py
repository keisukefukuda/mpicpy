import argparse
import hashlib
import os.path
import re
import sys
import time

from mpi4py import MPI

comm = MPI.COMM_WORLD


def die(msg=None):
    comm = MPI.COMM_WORLD
    if msg not in [None, '']:
        sys.stderr.write("-------------------------------------------------\n")
        sys.stderr.write("mpicpy *** ERROR ***: (Rank {}): {}\n".format(comm.rank, msg))
        sys.stderr.write("-------------------------------------------------\n")
        sys.stderr.flush()
    comm.barrier()
    comm.Abort()


def calc_md5(filepath):
    m = hashlib.md5()
    if os.path.exists(filepath):
        with open(filepath, mode='rb') as f:
            m.update(f.read())
        return m.hexdigest()
    else:
        return None


def max_with_index(lst):
    return max(enumerate(lst), key=lambda x: x[1])


def determine_root_rank(comm, args):
    """Determine which rank shall be the root.

    Availablle options:
      * size: The rank that has the largest file in size becomes the rank
      * md5:  The rank that has the file of the specified md5 becomes the rank.
              The md5 checksum is specified after the 'md5' option, separated by
              a separator ':' or '/'.
      * host: The first rank that is on the specified host becomes the rank.
              host:host1.
      * rank [digits]: specified rank becomes the root

    """

    filepath = args.filepath

    check = [e is not None for e in
             [args.size, args.md5, args.mtime, args.rank, args.hostname]]

    if check.count(True) != 1:
        if comm.rank == 0:
            die("One of --size, --md5, --mtime, --rank or --hostname must be specified")
        else:
            die()

    comm.barrier()

    if args.rank:
        root = args.rank
        assert 0 <= root < comm.size

        if root == comm.rank:
            if not os.path.exists(filepath):
                die('Bcast root {} does not have the file {}'.format(root,
                                                                     filepath))
        return root

    if args.size:
        try:
            size = os.path.getsize(filepath)
        except FileNotFoundError:
            size = 0

        size_list = comm.allgather(size)
        rank, val = max_with_index(size_list)
        if rank == comm.rank:
            print("Rank {} is root: file size {}".format(rank, val))
        return rank

    if args.md5:
        specified_checksum = args.md5
        file_checksum: str = calc_md5(filepath)
        md5_match_list = comm.allgather(file_checksum.find(specified_checksum) == 0)

        if all(not e for e in md5_match_list):
            if comm.rank == 0:
                die("No rank has a file of MD5 '{}'".format(args.md5))
            else:
                die()

        if all(e is False for e in md5_match_list):
            if comm.rank == 0:
                die("No rank has a file of checksum {}")
            else:
                die()

        root = md5_match_list.index(True)
        return root

    if args.hostname:
        specified_hostname = args.hostname
        local_hostname = os.uname()[1]
        match_list = comm.allgather(specified_hostname == local_hostname)
        if match_list.count(True) == 0:
            if comm.root == 0:
                die("No such hostname: {}".format(specified_hostname))
            else:
                die()
        elif match_list.count(True) > 1:
            if comm.root == 0:
                die('Multiple ranks have the same hostname: "{}"'.format(specified_hostname))
            else:
                die()
        else:
            # OK
            return match_list.index(True)

    if args.mtime:
        raise RuntimeError('Not implemented')

    assert False


def parse_chunk_size(s):
    if type(s) == int:
        assert s >= 0
        return s

    m = re.match(r'^(\d+?)(([kmg])i?b?)?$', s, re.I)

    if m is None:
        raise RuntimeError('Cannot parse chunk size: "{}"'.format(s))

    digits = int(m.group(1))

    suffix = (m.group(3) or '').lower()

    if suffix == 'k':
        digits *= 1024
    elif suffix == 'm':
        digits *= 1024 * 1024
    elif suffix == 'g':
        digits *= 1024 * 1024 * 1024
    elif suffix == '':
        pass
    else:
        assert False

    return digits


def send_file(filepath, chunk_size):
    size = os.path.getsize(filepath)

    assert size >= 0

    print("Rank {} [Root] size={}".format(comm.rank, size))
    comm.bcast(size, root=comm.rank)

    if size == 0:
        num_chunks = 0
    else:
        num_chunks = ((size - 1) // chunk_size) + 1

    print("num_chunks = {}".format(num_chunks))

    with open(filepath, 'rb') as f:
        for i in range(num_chunks):
            print("Sending Chunk #{}".format(i+1))
            buf = f.read(chunk_size)
            comm.Bcast(buf, root=comm.rank)


def recv_file(root, filepath, chunksize):
    size = None
    size = comm.bcast(size, root=root)
    print("\t\tRank {} size={}".format(comm.rank, size))

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


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('filepath', type=str,
                        help='Target file path')
    parser.add_argument('--size', action='store_true', default=None,
                        help="Determine root rank using largest file size")
    parser.add_argument('--md5', type=str, default=None,
                        help="Determine root rank using MD5 checksum")
    parser.add_argument('--mtime', action='store_true', default=None,
                        help="Determine root based on mtime")
    parser.add_argument('--rank', type=int, default=None,
                        help='Specify root rank')
    parser.add_argument('--hostname', type=str,
                        help='Specify hostname to be the root process')

    parser.add_argument('-f', '--force-overwrite', action='store_true', default=False,
                        help='Allow overriding existing file')
    parser.add_argument('-c', '--chunk-size', type=str, default='1GB',
                        help='Chunk size')
    parser.add_argument('--checksum', action='store_true', default=True,
                        help='Checksum after copy')
    parser.add_argument('--rank-suffix', action='store_true', default=False,
                        help='Add rank to the filename as suffix (for debugging)')

    args = parser.parse_args()

    if comm.size == 1:
        print("This program is useless with COMM_SIZE == 1")
        exit(0)

    root = determine_root_rank(comm, args)
    filepath = args.filepath

    assert type(root) == int
    assert 0 <= root < comm.size

    local_filename = filepath
    if args.rank_suffix:
        if comm.rank != root:
            local_filename = '{}.{}'.format(filepath, comm.rank)

    err = False
    if comm.rank != root and os.path.exists(local_filename):
        if not args.force_overwrite:
            err = True
            sys.stderr.write("*** ERROR *** Rank [{}]: File '{}' already exists.\n".format(comm.rank, local_filename))
            sys.stderr.flush()
    comm.barrier()
    if err:
        comm.Abort()

    comm.barrier()

    chunk_size = parse_chunk_size(args.chunk_size)

    if comm.rank == root:
        send_file(filepath, chunk_size)
    else:
        recv_file(root, local_filename, chunk_size)

    # calc checksum
    if args.checksum:
        checksum = calc_md5(local_filename)
        checksum_list = comm.gather(checksum, root=0)

        for i in range(comm.size):
            if i == comm.rank:
                print('Rank {}: MD5 {}'.format(comm.rank, checksum))
                sys.stdout.flush()
            comm.barrier()
        comm.barrier()

        if comm.rank == 0:
            if len(set(checksum_list)) != 1:
                for rank in range(comm.size):
                    sys.stderr.write("Rank {}: {}\n".format(rank, checksum_list[rank]))
                    sys.stderr.flush()
                die("Error: MD5 checksum mismatch: ")
            else:
                time.sleep(1)
                print("Checksum OK.")
                sys.stdout.flush()


if __name__ == '__main__':
    main()