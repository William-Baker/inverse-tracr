#!/gpfs/gibbs/pi/support/software/utilities/bin/python

import argparse
import subprocess
import sys

import numpy as np
import pandas as pd

from io import StringIO
import os

#import termplotlib as tpl

__version__ = 0.4
debug = False


# def time_to_float(time):
#     return days + hours + mins + secs

#@profile
def job_eff(job_id=0, cluster=os.getenv('SLURM_CLUSTER_NAME')):
    print("--------------------------------------------------------")

if __name__ == "__main__":

    desc = (
        """
    seff-array v%s
    https://github.com/ycrc/seff-array
    ---------------
    An extension of the Slurm command 'seff' designed to handle job arrays and display information in a histogram.

    To use seff-array on the job array with ID '12345678', simply run 'seff-array 12345678'.

    Other things can go here in the future.
    -----------------
    """
        % __version__
    )

    parser = argparse.ArgumentParser(
        formatter_class=argparse.RawDescriptionHelpFormatter,
        description=desc,
    )
    parser.add_argument("jobid")
    parser.add_argument("-c", "--cluster", action="store", dest="cluster")
    parser.add_argument('--version', action='version',  version='%(prog)s {version}'.format(version=__version__))
    args = parser.parse_args()

    job_eff(args.jobid, args.cluster)