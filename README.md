# mpicpy
Broadcast files using MPI via high speed interconnect (if you have :)

## Install

```
pip install mpicpy
```

## Basic Usage
Mpicpy broadcasts a single file from one of the MPI ranks to the others.


### Specifying the root.

There may be files of the same name on multiple hosts, possibly because
of a failed file transfer session in the past.
You can specify which file  

There are four ways to determine the root process: `size`, `md5`, 'rank', and `mtime`.
(NOTE: currently `mtime` is not supported yet.)

 * `--auto`: (default) Makes sure if only one process has the specified file and transfer it.
 * `--size`: If multiple ranks have the same filename, the file of the largest size becomes the original.
 * `--md5=XXXXX`: The file of which MD5 checksum is XXXXX.
 * `--rank=RANK`: The specified MPI rank becomes the root process.
 * `--mtime`: The file that has the newest mtime becomes the original.
 
