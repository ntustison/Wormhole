
import ants
import antspynet
import sys

import os.path
from os import path
from shutil import copyfile

import tensorflow as tf

t1_file = sys.argv[1]
output_prefix = sys.argv[2]
threads = int(sys.argv[3])

tf.keras.backend.clear_session()
config = tf.compat.v1.ConfigProto(intra_op_parallelism_threads=threads,
                                  inter_op_parallelism_threads=threads)
session = tf.compat.v1.Session(config=config)
tf.compat.v1.keras.backend.set_session(session)

t1 = ants.image_read(t1_file)

dkt = None
dkt_file = output_prefix + "_ants_dkt.nii.gz"
if not path.exists(dkt_file):
    print("    Calculating\n")
    dkt = antspynet.desikan_killiany_tourville_labeling(t1, verbose=True)
    ants.image_write(dkt, dkt_file)
else:
    dkt = ants.image_read(dkt_file)


dkt_lobes_file = output_prefix + "_ants_dktLobes.nii.gz"
if not path.exists(dkt_lobes_file):
    print("    Calculating\n")
    dkt_lobes = antspynet.dkt_based_lobar_parcellation(t1, dkt = dkt, verbose=True)
    ants.image_write(dkt_lobes, dkt_lobes_file)

