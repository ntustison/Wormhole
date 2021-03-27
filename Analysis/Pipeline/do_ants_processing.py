
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

wmh_file = output_prefix + "_ants_FLAIR_SysuWmhSegmentation.nii.gz"
if not path.exists(wmh_file):
    print("    Calculating\n")
    wmh = antspynet.sysu_media_wmh_segmentation(flair, t1=t1, do_preprocessing=True, use_ensemble=True, use_axial_slices_only=False, verbose=True)
    wmh_seg = ants.threshold_image(wmh, 0.5, 1, 1, 0)
    ants.image_write(wmh_seg, wmh_file)

wmh_file = output_prefix + "_ants_FLAIR_SysuWmhAxialSegmentation.nii.gz"
if not path.exists(wmh_file):
    print("    Calculating\n")
    wmh = antspynet.sysu_media_wmh_segmentation(flair, t1=t1, do_preprocessing=True, use_ensemble=True, use_axial_slices_only=True, verbose=True)
    wmh_seg = ants.threshold_image(wmh, 0.5, 1, 1, 0)
    ants.image_write(wmh_seg, wmh_file)

wmh_file = output_prefix + "_ants_FLAIR_EwDavidOctantWmhSegmentation.nii.gz"
if not path.exists(wmh_file):
    print("    Calculating\n")
    wmh = antspynet.ew_david(flair, t1=t1, do_preprocessing=True, do_slicewise=False, verbose=True)
    wmh_seg = ants.threshold_image(wmh, 0.5, 1, 1, 0)
    ants.image_write(wmh_seg, wmh_file)

wmh_file = output_prefix + "_ants_FLAIR_EwDavidSlicewiseWmhSegmentation_300mb.nii.gz"
if not path.exists(wmh_file):
    print("    Calculating\n")
    wmh = antspynet.ew_david(flair, t1=t1, do_preprocessing=True, do_slicewise=True, verbose=True)
    wmh_seg = ants.threshold_image(wmh, 0.5, 1, 1, 0)
    ants.image_write(wmh_seg, wmh_file)

# wmh_file = output_prefix + "_ants_FLAIR_EwDavidSlicewiseWmhSegmentation_7mb.nii.gz"
# if not path.exists(wmh_file):
#     print("    Calculating\n")
#     wmh = antspynet.ew_david(flair, t1=t1, do_preprocessing=True, do_slicewise=True, verbose=True)
#     wmh_seg = ants.threshold_image(wmh, 0.5, 1, 1, 0)
#     ants.image_write(wmh_seg, wmh_file)

