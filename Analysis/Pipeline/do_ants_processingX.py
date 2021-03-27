
import ants
import antspynet
import sys

import os.path
from os import path
from shutil import copyfile

import tensorflow as tf

t1_file = sys.argv[1]
flair_file = sys.argv[2]
output_prefix = sys.argv[3]
threads = int(sys.argv[4])

tf.keras.backend.clear_session()
config = tf.compat.v1.ConfigProto(intra_op_parallelism_threads=threads,
                                  inter_op_parallelism_threads=threads)
session = tf.compat.v1.Session(config=config)
tf.compat.v1.keras.backend.set_session(session)

t1 = ants.image_read(t1_file)
flair = ants.image_read(flair_file)

print("T1xFLAIR registration")

t1xflair_file = output_prefix + "_ants_T1xFLAIR.nii.gz"
t1xflair_xfrm = output_prefix + "_ants_T1xFLAIR0GenericAffine.mat"
t1xflair = None
if not path.exists(t1xflair_file) or not path.exists(t1xflair_xfrm):
    print("    Calculating\n")
    reg = ants.registration(t1, flair, type_of_transform="antsRegistrationSyNQuick[r]")
    t1xflair = reg["warpedmovout"]
    ants.image_write(t1xflair, t1xflair_file)
    copyfile(reg['fwdtransforms'][0], t1xflair_xfrm)
else:
    print("    Reading\n")
    t1xflair = ants.image_read(t1xflair_file)

print("Atropos\n")

atropos_file = output_prefix + "_ants_BrainSegmentation.nii.gz"
if not path.exists(atropos_file):
    print("    Calculating\n")
    atropos = antspynet.deep_atropos(t1, do_preprocessing=True, verbose=True)
    atropos_segmentation = atropos['segmentation_image']
    ants.image_write(atropos_segmentation, atropos_file)

print("SysuWMH")

wmh_file = output_prefix + "_ants_SysuWmhSegmentation.nii.gz"
if not path.exists(wmh_file):
    print("    Calculating\n")
    wmh = antspynet.sysu_media_wmh_segmentation(t1xflair, t1=t1, do_preprocessing=True, use_ensemble=True, use_axial_slices_only=False, verbose=True)
    wmh_seg = ants.threshold_image(wmh, 0.5, 1, 1, 0)
    ants.image_write(wmh_seg, wmh_file)

