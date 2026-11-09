"""Console script `3d-ice-emulator`: run the bundled binary with the same arguments."""
import os
import subprocess
import sys

from . import executable


def main(argv=None):
    exe = executable()
    args = [exe] + list(sys.argv[1:] if argv is None else argv)
    if os.name == "posix":
        os.execv(exe, args)                        # replaces this process; exit code and signals pass through
    return subprocess.call(args)


if __name__ == "__main__":
    sys.exit(main())
