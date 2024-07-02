import os
import sys
import pyedflib
import pandas as pd
from io import StringIO
import re
import numpy as np
from os import listdir
from fnmatch import fnmatch
import shutil
import time
from tqdm import tqdm
import time

import preprocessing.nedc_debug_tools as ndt
import preprocessing.nedc_edf_tools as net
import preprocessing.nedc_file_tools as nft
import preprocessing.nedc_mont_tools as nmt



#------------------------------------------------------------------------------
#
# global variables are listed here
#
#------------------------------------------------------------------------------

# define default argument valuesp
#
DEF_BSIZE = int(10)
DEF_FORMAT_FLOAT = "float"
DEF_FORMAT_SHORT = "short"
DEF_MODE = False

#------------------------------------------------------------------------------
#
# functions are listed here
#
#------------------------------------------------------------------------------

# declare a global debug object so we can use it in functions
#
dbgl = ndt.Dbgl()
dbgl.set(level = 1, name = 'BRIEF')
# function: nedc_pystream_edf
#
# arguments:
#  fname: filename to be processed
#  montage: a montage object to be used for processing
#  bsize: the block size to be used for printing
#  format: print as floats or short ints
#  mode: ASCII is false, binary is true
#  fp: an open file pointer
#
# return: a boolean indicating status
#
def nedc_pystream_edf(fname, montage, bsize, format, mode, output_fname,fp = sys.stdout):

    # declare local variables
    #
    edf = net.Edf()

    # display an informational message
    #
    if dbgl > ndt.BRIEF:
        fp.write("%s (line: %s) %s: streaming the signal (%s)\n" %
                 (__FILE__, ndt.__LINE__, ndt.__NAME__, fname))

    # expand the filename (checking for environment variables)
    #
    ffile = nft.get_fullpath(fname)

    # read the unscaled Edf signal
    #
    if (format == DEF_FORMAT_SHORT):
        (h, isig) = edf.read_edf(ffile, False, True)
        if  isig == None:
            fp.write("Error: %s (line: %s) %s: %s\n" %
                     (__FILE__, ndt.__LINE__, ndt.__NAME__,
                      "error reading signal as short ints"))
            return False
    else:
        (h, isig) = edf.read_edf(ffile, True, True)
        if  isig == None:
            fp.write("Error: %s (line: %s) %s: %s\n" %
                     (__FILE__, ndt.__LINE__, ndt.__NAME__,
                      "error reading signal as floats"))
            return False

    # apply the montage
    #
    #
    mnt = nmt.Montage()
    osig = mnt.apply(isig, montage);
    if  osig == None:
        fp.write("Error: %s (line: %s) %s: %s\n" %
                 (__FILE__, ndt.__LINE__, ndt.__NAME__,
                  "error applying montage"))
        return False
    bool_to_file = output_fname != ""
    # case: ascii mode
    #
    if mode is False:
    
        # get the number of samples per channel
        #
        key = next(iter(osig))
        nchannels = len(osig)
        i = int(0);
        iframe = int(0);
        iend = len(osig[key])
        signals = []
        sheaders = []
        sfreq = 250
        # display some useful signal information
        #
        if dbgl > ndt.NONE:
            fp.write("file name %s\n" % output_fname)
            fp.write("number of output channels = %d\n" % (nchannels))
            fp.write("number of samples per channel = %d\n" % (iend))

        for key in osig:
            sfreq = abs(max(osig[key], key=abs))
            # print("key: ", key," sfreq: ", sfreq)
            shead = pyedflib.highlevel.make_signal_header(f'{key}', physical_min=-sfreq ,physical_max=sfreq)
            sheaders.append(shead)
            signals.append(osig[key])
            pyedflib.highlevel.write_edf(output_fname, signals=signals, signal_headers=sheaders)
    # case: binary mode
    #
    else:

        # get the number of samples per channel
        #
        key = next(iter(osig))
        nchannels = len(osig)
        i = int(0);
        iframe = int(0);
        iend = len(osig[key])

        # display some useful signal information
        #
        if dbgl > ndt.NONE:
            fp.write("mode is binary\n")
            fp.write("number of output channels = %d\n" % (nchannels))
            fp.write("number of samples per channel = %d\n" % (iend))

        # do a vector based conversion of the data for speed
        #
        if format == DEF_FORMAT_SHORT:
            tmp = {}
            for key in osig:
                tmp[key] = np.clip(osig[key],
                                   DEF_SHORT_MINVAL, DEF_SHORT_MAXVAL)
                tmp[key].round()

        # loop over the samples
        #
        while i < iend:

            # display some information to make the output more readable
            #
            if dbgl > ndt.NONE:

                # display frame information
                #
                fp.write("frame %5d: sample %8d to %8d\n" % (iframe, i, i + bsize))

                # display montage labels
                #
                fp.write("%8s " % (nft.STRING_EMPTY))
                for key in osig:
                    fp.write("%10s " % (key))
                fp.write(nft.DELIM_NEWLINE)

            # display the sample values: write one channel at a time
            #
            for key in osig:
                for j in range(i, i + bsize):

	            # make sure we don't exceed the signal length
	            #
                    if j < iend:

                        # fmt == DEF_FORMAT_FLOAT: write binary data as floats
                        #  note that the signal is a double. note also these
                        #  lines don't really fit in our 80-col format :(
                        #
                        if format == DEF_FORMAT_FLOAT:
                            sys.stdout.buffer.write(struct.pack('<f',
                                                                osig[key][j]))
                        else:
                            sys.stdout.buffer.write(struct.pack('<h',
                                                                int(tmp[key][j])))

            # increment counters
            #
            i += bsize
            iframe += int(1)

    # exit gracefully
    #
    return True

montage_files = {
'01_tcp_ar':'montages/01_tcp_ar_montage.txt',
'02_tcp_le': 'montages/02_tcp_le_montage.txt',
'03_tcp_ar_a': 'montages//03_tcp_ar_a_montage.txt',
'04_tcp_le_a': 'montages//04_tcp_le_a_montage.txt'
}

def extract_filename_part(file_path):
    match = re.search(r'([^/]+)\.edf$', file_path)
    if match:
        return match.group(0)
    return None

def process(edf_fp, montage):
    # Extract montage filename
    montage_fname = montage_files.get(montage, 'NA')
    mnt = nmt.Montage()

    # Load montage object if available
    if montage != 'NA':
        montage_obj = mnt.load(montage_fname)
    else:
        return None
    
    # Assuming `dev` and other variables are defined elsewhere
    output_edf_fname = "data/" + extract_filename_part(edf_fp)
    
    # Copy EDF file to output directory
    shutil.copy(edf_fp, output_edf_fname)

    # Process EDF file
    print(time.time(), "<-----Start processing: ", edf_fp, "---------->\n")
    nedc_pystream_edf(edf_fp, montage_obj, DEF_BSIZE, DEF_FORMAT_FLOAT, DEF_MODE, output_edf_fname, sys.stdout)
    print(time.time(), "<-----Finished processing: ", extract_filename_part(edf_fp), "---------->\n")
    return output_edf_fname


