# Copyright (c) Genome Research Ltd 2012
# Copyright (c) Universite Laval 2018
# Author
# Guy Coates <gmpc@sanger.ac.uk>
# Simon Guilbault <simon.guilbault@calculquebec.ca>
# This program is released under the GNU Public License V2 (GPLv2)
# Based on https://github.com/wtsi-ssg/pcp/blob/master/pcplib/lustreapi.py

"""
Python bindings to minimal subset of lustre api.
This module requires a dynamically linked version of the lustre
client library (liblustreapi.so).

Older version of the lustre client only ships a static library
(liblustreapi.a).
setup.py should have generated a dynamic version during installation.

You can generate the dynamic library by hand by doing the following:

ar -x liblustreapi.a
gcc -shared -o liblustreapi.so *.o

"""
import ctypes
import ctypes.util
import os
import select


import pkg_resources
try:
    __version__ = pkg_resources.require("pcp")[0].version
except pkg_resources.DistributionNotFound:
    __version__ = "UNRELEASED"

LUSTREMAGIC = 0xbd00bd0

liblocation = ctypes.util.find_library("lustreapi")
# See if liblustreapi.so is in the same directory as the module
if not liblocation:
    modlocation, module = os.path.split(__file__)
    liblocation = os.path.join(modlocation, "liblustreapi.so")

glibc = ctypes.CDLL("libc.so.6", use_errno=True)
lustre = ctypes.CDLL(liblocation, use_errno=True)

# ctype boilerplate for C data structures and functions
class lov_user_ost_data_v1(ctypes.Structure):
    _fields_ = [
        ("l_object_id", ctypes.c_ulonglong),
        ("l_object_seq", ctypes.c_ulonglong),
        ("l_ost_gen", ctypes.c_uint),
        ("l_ost_idx", ctypes.c_uint)
    ]


class lov_user_md_v1(ctypes.Structure):
    _fields_ = [
        ("lmm_magic", ctypes.c_uint),
        ("lmm_pattern", ctypes.c_uint),
        ("lmm_object_id", ctypes.c_ulonglong),
        ("lmm_object_seq", ctypes.c_ulonglong),
        ("lmm_stripe_size", ctypes.c_uint),
        ("lmm_stripe_count",  ctypes.c_short),
        ("lmm_stripe_offset", ctypes.c_short),
        ("lmm_objects", lov_user_ost_data_v1 * 2000),
    ]

    def __str__(self) -> str:
        return "lov_user_md_v1: lmm_magic=%d, lmm_pattern=%d, lmm_object_id=%d, lmm_object_seq=%d, lmm_stripe_size=%d, lmm_stripe_count=%d, lmm_stripe_offset=%d, lmm_objects=%s" % (
            self.lmm_magic, self.lmm_pattern, self.lmm_object_id, self.lmm_object_seq, self.lmm_stripe_size, self.lmm_stripe_count, self.lmm_stripe_offset, self.lmm_objects)

class Stat(ctypes.Structure):
    _fields_ = [
        ("st_dev", ctypes.c_ulong),     # Device ID
        ("st_ino", ctypes.c_ulong),     # Inode number
        ("st_mode", ctypes.c_uint),     # File mode
        ("st_nlink", ctypes.c_uint),    # Number of hard links
        ("st_uid", ctypes.c_uint),      # User ID of owner
        ("st_gid", ctypes.c_uint),      # Group ID of owner
        ("st_rdev", ctypes.c_ulong),    # Device type (if inode device)
        ("st_size", ctypes.c_long),     # Total size, in bytes
        ("st_blksize", ctypes.c_long),  # Block size for filesystem I/O
        ("st_blocks", ctypes.c_long),   # Number of blocks allocated
        ("st_atime", ctypes.c_long),    # Time of last access
        ("st_atimensec", ctypes.c_long),# Nanoseconds of last access
        ("st_mtime", ctypes.c_long),    # Time of last modification
        ("st_mtimensec", ctypes.c_long),# Nanoseconds of last modification
        ("st_ctime", ctypes.c_long),    # Time of last status change
        ("st_ctimensec", ctypes.c_long) # Nanoseconds of last status change
    ]

class LovUserMdsDataV1(ctypes.Structure):
    _pack_ = 1  # Ensures the structure is packed, matching the __attribute__((packed))
    _fields_ = [
        ("lmd_st", Stat),              # Using the defined Stat structure
        ("lmd_lmm", lov_user_md_v1),      # Using the LovUserMdV1 structure
    ]

LOV_MAXPOOLNAME = 16
UUID_MAX = 40

class ObdUuid(ctypes.Structure):
    # Define the fields of struct obd_uuid if necessary
    _fields_ = [
        ("uuid", ctypes.c_char * UUID_MAX),
    ]

LovUserMdsData = lov_user_md_v1

class FindParam(ctypes.Structure):
    _fields_ = [
        ("maxdepth", ctypes.c_uint),
        ("atime", ctypes.c_long),     # time_t is typically represented as long
        ("mtime", ctypes.c_long),
        ("ctime", ctypes.c_long),
        ("asign", ctypes.c_int),
        ("csign", ctypes.c_int),
        ("msign", ctypes.c_int),
        ("type", ctypes.c_int),
        ("size", ctypes.c_ulonglong),
        ("size_sign", ctypes.c_int),
        ("size_units", ctypes.c_ulonglong),
        ("uid", ctypes.c_uint),       # uid_t is typically an unsigned int
        ("gid", ctypes.c_uint),       # gid_t is typically an unsigned int
        ("zeroend", ctypes.c_ulong, 1),
        ("recursive", ctypes.c_ulong, 1),
        ("got_uuids", ctypes.c_ulong, 1),
        ("obds_printed", ctypes.c_ulong, 1),
        ("exclude_pattern", ctypes.c_ulong, 1),
        ("exclude_type", ctypes.c_ulong, 1),
        ("exclude_obd", ctypes.c_ulong, 1),
        ("have_fileinfo", ctypes.c_ulong, 1),
        ("exclude_gid", ctypes.c_ulong, 1),
        ("exclude_uid", ctypes.c_ulong, 1),
        ("check_gid", ctypes.c_ulong, 1),
        ("check_uid", ctypes.c_ulong, 1),
        ("check_pool", ctypes.c_ulong, 1),
        ("check_size", ctypes.c_ulong, 1),
        ("exclude_pool", ctypes.c_ulong, 1),
        ("exclude_size", ctypes.c_ulong, 1),
        ("exclude_atime", ctypes.c_ulong, 1),
        ("exclude_mtime", ctypes.c_ulong, 1),
        ("exclude_ctime", ctypes.c_ulong, 1),
        ("get_mdt_index", ctypes.c_ulong, 1),
        ("raw", ctypes.c_ulong, 1),
        ("verbose", ctypes.c_int),
        ("quiet", ctypes.c_int),
        ("pattern", ctypes.c_char_p),
        ("print_fmt", ctypes.c_char_p),
        ("obduuid", ctypes.POINTER(ObdUuid)),
        ("num_obds", ctypes.c_int),
        ("num_alloc_obds", ctypes.c_int),
        ("obdindex", ctypes.c_int),
        ("obdindexes", ctypes.POINTER(ctypes.c_int)),
        ("lumlen", ctypes.c_int),
        ("lmd", ctypes.POINTER(LovUserMdsDataV1)),
        ("depth", ctypes.c_uint),
        ("st_dev", ctypes.c_ulong),   # dev_t is typically an unsigned long
        ("poolname", ctypes.c_char * (LOV_MAXPOOLNAME + 1)),
    ]

    # def __str__(self) -> str:
    #     return "FindParam: maxdepth=%d, atime=%d, mtime=%d, ctime=%d, asign=%d, csign=%d, msign=%d, type=%d, size=%d, size_sign=%d, size_units=%d, uid=%d, gid=%d, zeroend=%d, recursive=%d, got_uuids=%d, obds_printed=%d, exclude_pattern=%d, exclude_type=%d, exclude_obd=%d, have_fileinfo=%d, exclude_gid=%d, exclude_uid=%d, check_gid=%d, check_uid=%d, check_pool=%d, check_size=%d, exclude_pool=%d, exclude_size=%d, exclude_atime=%d, exclude_mtime=%d, exclude_ctime=%d, get_mdt_index=%d, raw=%d, verbose=%d, quiet=%d, pattern=%s, print_fmt=%s, obduuid=%s, num_obds=%d, num_alloc_obds=%d, obdindex=%d, obdindexes=%s, lumlen=%d, lmd=%s, depth=%d, st_dev=%d, poolname=%s" % (
    #             self.maxdepth, self.atime, self.mtime, self.ctime, self.asign, self.csign, self.msign, self.type, self.size, self.size_sign, self.size_units, self.uid, self.gid, self.zeroend, self.recursive, self.got_uuids, self.obds_printed, self.exclude_pattern, self.exclude_type, self.exclude_obd, self.have_fileinfo, self.exclude_gid, self.exclude_uid, self.check_gid, self.check_uid, self.check_pool, self.check_size, self.exclude_pool, self.exclude_size, self.exclude_atime, self.exclude_mtime, self.exclude_ctime, self.get_mdt_index, self.raw, self.verbose, self.quiet, self.pattern, self.print_fmt, self.obduuid, self.num_obds, self.num_alloc_obds, self.obdindex, self.obdindexes, self.lumlen, self.lmd, self.depth, self.st_dev, self.poolname) 


    @staticmethod
    def zero_init():
        param = FindParam()
        param.maxdepth = 1
        return param


lustre.llapi_getstripe.argtypes = [ctypes.c_char_p, ctypes.POINTER(FindParam)]
lustre.llapi_getstripe.restype = ctypes.c_int

lustre.llapi_file_create_pool.argtypes = [ctypes.c_char_p, ctypes.c_ulonglong,
                                          ctypes.c_int, ctypes.c_int, ctypes.c_int,
                                          ctypes.c_char_p]
lustre.llapi_file_create_pool.restype = ctypes.c_int

class capture_stderr():
    """This class intercepts stderr and stores any output"""
    def __init__(self):
        self.pipeout, self.pipein = os.pipe()
        self.oldstderr = os.dup(2)
        os.dup2(self.pipein, 2)
        self.contents = ""

    def __str__(self):
        return (self.contents)

    def read_data(self):
        """Read data from stderr until there is no more."""
        while self.check_data():
            self.contents += os.read(self.pipeout, 1024)

    def check_data(self):
        """Check to see if there is any data to be read."""
        r, _, _ = select.select([self.pipeout], [], [], 0)
        return bool(r)

    def stop_capture(self):
        """Restore the original stderr"""
        os.dup2(self.oldstderr, 2)
        os.close(self.pipeout)
        os.close(self.pipein)

def getstripe(filename: str):
    """Returns a stripeObj containing the stipe information of filename.

    Arguments:
      filename: The name of the file to query.

    Returns:
      A stripeObj containing the stripe information.
    """
    param = FindParam()
    param.maxdepth = 1

    # message = capture_stderr()
    rc = lustre.llapi_getstripe(filename.encode(), ctypes.byref(param))
    # message.read_data()
    # message.stop_capture()

    # err 61 is due to  LU-541 (see below)
    if rc < 0 or rc > 0:
        raise Exception("Error %d" % rc)

    return rc


def setstripe(filename: str, stripesize=0, stripeoffset=-1, stripecount=1):
    """Sets the striping on an existing directory, or create a new empty file
    with the specified striping. Stripe parameters can be set explicity, or
    you can pass in an existing stripeobj to copy the attributes from an
    existing file.

    Note you can set the striping on an existing directory, but you cannot set
    the striping on an existing file.

    Arguments:
      stripesize: size of stripe in bytes
      stripeoffset: stripe offset
      stripecount: stripe count

    Examples:
      #Set the filesystem defaults
      setstripe("/lustre/testfile")

      # Stripe across all OSTs.
      setstripe("/lustre/testfile", stripecount=-1)

      #copy the attributes from foo
      stripeobj = getstripe("/lustre/foo")
      setstripe("/lustre/testfile", stripeobj)

    """
    print("Setting stripe count to %d" % stripecount)

    fd = lustre.llapi_file_create_pool(filename.encode(), stripesize, stripeoffset, stripecount, 0, None)

    if fd < 0:
        err = 0 - fd
        raise IOError(err, os.strerror(err))
    else:
        os.close(fd)
        return 0


if __name__ == "__main__":
    # obj = getstripe("./data/stormer/train/img_0000_of_2920.hdf5")
    # print(obj)
    path = "/lus/eagle/projects/MDClimSim/rayandrew/dlio-benchmark/test-data/"
    os.makedirs(path, exist_ok=True)
    setstripe(path, stripecount=-1)
    obj = getstripe(path)
    print(str(obj))
