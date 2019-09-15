import os.path
import sys

import pytest

project_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(project_dir)

from mpicpy import *


def test_parse_chunk_size():
    for s in ['1g', '1gb', '1gib', '1GiB', '1GB']:
        assert parse_chunk_size(s) == 1024 * 1024 * 1024

    for s in ['1m', '1mb', '1mib', '1MiB', '1MB']:
        assert parse_chunk_size(s) == 1024 * 1024

    for s in ['1k', '1kb', '1kib', '1KiB', '1KB']:
        assert parse_chunk_size(s) == 1024

    for s in ['10']:
        assert parse_chunk_size(s) == 10


def test_md5():
    LICENSE_md5 = 'e18891f7e7107c7e5861dfffa1668916'
    assert calc_md5(os.path.join(project_dir, 'LICENSE')) == LICENSE_md5


def test_get_num_chunks():
    assert get_num_chunks(file_size=0, chunk_size=10) == 0
    assert get_num_chunks(file_size=300, chunk_size=120) == 3
    assert get_num_chunks(file_size=1000, chunk_size=1) == 1000
