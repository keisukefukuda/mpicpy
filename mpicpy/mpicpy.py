import argparse
import hashlib
import os.path
import platform
import re
import sys
import time

from tqdm import tqdm

MPICPY_ERR_GENEREAL = 1
MPICPY_ERR_FILE_ALREADY_EXISTS = 2


from mpi4py import MPI


def get_local_rank(comm):
    if 'OMPI_COMM_WORLD_LOCAL_RANK' in os.environ and comm == MPI.COMM_WORLD:
        return int(os.environ.get('OMPI_COMM_WORLD_LOCAL_RANK'))

    hostname = platform.uname()[1]

    hostnames = comm.allgather(hostname)
    local_rank = hostnames[:comm.rank].count(hostname)

    return local_rank


def log_label(comm):
    return "mpicpy: (Rank {},{}): ".format(comm.rank, MPI.COMM_WORLD.rank)


def mpi_print(comm, msg, out=sys.stdout):
    for i in range(comm.size):
        if comm.rank == i:
            out.write(log_label(comm) + msg + "\n")
            out.flush()
        comm.barrier()


def die(msg=None, errcode=MPICPY_ERR_GENEREAL):
    comm = MPI.COMM_WORLD
    if msg not in [None, '']:
        sys.stderr.write("-------------------------------------------------\n")
        sys.stderr.write("mpicpy *** ERROR ***: (Rank {}): {}\n".format(comm.rank, msg))
        sys.stderr.write("-------------------------------------------------\n")
        sys.stderr.flush()
    comm.Abort(errcode)


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


def determine_root_rank(comm, filepath, args):
    """Determine which rank shall be the root.

    Availablle options:
      * auto: checks if only one rank has the ran file and broadcast it.
      * size: The rank that has the largest file in size becomes the rank
      * md5:  The rank that has the file of the specified md5 becomes the rank.
              The md5 checksum is specified after the 'md5' option, separated by
              a separator ':' or '/'.
      * host: The first rank that is on the specified host becomes the rank.
              host:host1.
      * rank [digits]: specified rank becomes the root

    """

    check = [e is not None for e in
             [args.size, args.md5, args.mtime, args.rank, args.hostname]]

    if check.count(True) > 1:
        if comm.rank == 0:
            die("One of --auto, --size, --md5, --mtime, --rank or --hostname must be specified")
        else:
            die()

    comm.barrier()

    if check.count(True) == 0:
        # None of --auto, --size, --md5, --mtime, --rank --hostname is specified.
        # assume --auto (default)
        file_exists_mine = os.path.exists(filepath)
        file_exists_all = comm.allgather(file_exists_mine)
        if file_exists_all.count(True) == 0:
            die("No rank has the file to be broadcast")

        elif file_exists_all.count(True) > 1:
            die("--auto is specified but multiple ranks have the file")

        # Find the root
        root_rank = file_exists_all.index(True)
        assert 0 <= root_rank < comm.size
        return root_rank

    elif args.rank is not None:
        root = args.rank
        assert 0 <= root < comm.size

        if root == comm.rank:
            if not os.path.exists(filepath):
                die('Bcast root {} does not have the file {}'.format(root,
                                                                     filepath))
        return root

    elif args.size is not None:
        try:
            size = os.path.getsize(filepath)
        except FileNotFoundError:
            size = None

        size_list = comm.allgather(size)

        if all(s is None for s in size_list):
            die("No rank has the specified file")

        size_list = [s or 0 for s in size_list]

        rank, val = max_with_index(size_list)
        if rank == comm.rank:
            print(log_label(comm) + "Rank {} is root".format(rank, val))
        return rank

    elif args.md5 is not None:
        specified_checksum = args.md5
        assert type(specified_checksum) == str
        assert len(specified_checksum) > 0

        # TODO(keisukefukuda) .find() is a bug? check
        file_checksum = calc_md5(filepath) or ""
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

    elif args.hostname is not None:
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

    elif args.mtime is not None:
        raise RuntimeError('Not implemented')

    else:
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


def get_num_chunks(file_size, chunk_size):
    if file_size == 0:
        return 0
    else:
        return ((file_size - 1) // chunk_size) + 1


def show_file_info(comm, filepath, root):
    exists = os.path.exists(filepath)
    hostname = platform.uname()[1]
    if exists:
        size = os.path.getsize(exists)
    else:
        size = "N/A"

    root_indicator = '*' if root == comm.rank else ''

    msg = " {root:>1} [{host}] {size:>10} {filepath}".format(
        root=root_indicator,
        host=hostname,
        size=size,
        filepath=filepath)

    if comm.rank == 0:
        print('-' * len(msg), flush=True)
    mpi_print(comm, msg)
    if comm.rank == comm.size - 1:
        print('-' * len(msg), flush=True)


def send_file(comm, filepath, chunk_size):
    size = os.path.getsize(filepath)

    assert size >= 0

    # print("Rank {} [Root] size={}".format(comm.rank, size))
    comm.bcast(size, root=comm.rank)

    num_chunks = get_num_chunks(size, chunk_size)

    # print("num_chunks = {}".format(num_chunks))
    if size < 1024 * 1024:
        scale = lambda x: x
        unit = 'B'
    else:
        scale = lambda x: x / (1024.0 * 1024.)
        unit = 'MiB'

    bar_format = '{l_bar}{bar}| ' \
                 '{n_fmt}/{total_fmt} ' \
                 '[{elapsed}<{remaining}, {rate_fmt}{postfix}]'

    with tqdm(total=scale(size), bar_format=bar_format, unit=unit) as pbar:
        with open(filepath, 'rb') as f:
            for i in range(num_chunks):
                # print("Sending Chunk #{}".format(i+1))
                buf = f.read(chunk_size)
                comm.Bcast(buf, root=comm.rank)

                if i < num_chunks - 1:
                    sent_bytes = chunk_size
                else:
                    if size % chunk_size == 0:
                        sent_bytes = chunk_size
                    else:
                        sent_bytes = size % chunk_size
                pbar.update(scale(sent_bytes))


def recv_file(comm, root, filepath, chunk_size):
    size = None
    size = comm.bcast(size, root=root)
    # print("\t\tRank {} size={}".format(comm.rank, size))

    num_chunks = get_num_chunks(size, chunk_size)

    # print("\t\tnum_chunks = {}".format(num_chunks))

    with open(filepath, 'wb') as f:
        for i in range(num_chunks):
            if i < num_chunks - 1:
                buf = bytearray(chunk_size)
            else:
                if size % chunk_size == 0:
                    buf = bytearray(chunk_size)
                else:
                    buf = bytearray(size % chunk_size)

            # print("\t\tReceiving Chunk #{} from {}".format(i+1, root))
            comm.Bcast(buf, root=root)
            f.write(buf)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('filepath', type=str,
                        help='Target file path')

    # How to determine root?
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
    parser.add_argument('--auto', action='store_true',
                        help='Checks if only one host has the file and broadcast it.')

    # Additional options
    parser.add_argument('-f', '--force-overwrite', action='store_true', default=False,
                        help='Allow overriding existing file')
    parser.add_argument('-c', '--chunk-size', type=str, default='1GB',
                        help='Chunk size')
    parser.add_argument('--checksum', action='store_true', default=True,
                        help='Checksum after copy')
    parser.add_argument('--no-format-filename', action='store_true', default=None,
                        help="No format() application to filename")
    parser.add_argument('-o', '--allow-node-over-subscription',
                        default=False, action='store_true',
                        help="Allow multiple ranks on a single node.")

    args = parser.parse_args()

    if args.allow_node_over_subscription:
        comm = MPI.COMM_WORLD
    else:
        local_rank = get_local_rank(MPI.COMM_WORLD)
        comm = MPI.COMM_WORLD.Split(local_rank, MPI.COMM_WORLD.rank)
        if local_rank > 0:
            return

    if comm.size == 1:
        print("This program is useless with COMM_SIZE == 1")
        exit(0)

    filepath = args.filepath
    if not args.no_format_filename:
        filepath = filepath.format(rank=comm.rank)

    root = determine_root_rank(comm, filepath, args)

    assert type(root) == int
    assert 0 <= root < comm.size

    # TDOO: create a new communicator

    if comm.rank != root and os.path.exists(filepath):
        if not args.force_overwrite:
            die("File '{}' already exists.".format(filepath),
                errcode=MPICPY_ERR_FILE_ALREADY_EXISTS)
    comm.barrier()

    show_file_info(comm, filepath, root)

    chunk_size = parse_chunk_size(args.chunk_size)

    if comm.rank == root:
        send_file(comm, filepath, chunk_size)
    else:
        recv_file(comm, root, filepath, chunk_size)

    # calc checksum
    if args.checksum:
        checksum = calc_md5(filepath)
        checksum_list = comm.gather(checksum, root=0)

        for i in range(comm.size):
            if i == comm.rank:
                # print('Rank {}: MD5 {}'.format(comm.rank, checksum))
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
                print("\n" + log_label(comm) + "Checksum OK.")
                sys.stdout.flush()


if __name__ == '__main__':
    try:
        main()
    except Exception as e:
        sys.stderr.write("mpicpy: **** Error **** on Rank {}\n".format(comm.rank))

        import traceback
        traceback.print_exc(file=sys.stderr)
        sys.stderr.flush()

        MPI.COMM_WORLD.Abort(1)