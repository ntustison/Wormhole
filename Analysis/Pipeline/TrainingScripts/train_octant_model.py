import ants
import antspynet

import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
import glob

import random
import math

import tensorflow as tf
import tensorflow.keras as keras
import tensorflow.keras.backend as K

K.clear_session()
# gpus = tf.config.experimental.list_physical_devices("GPU")
# if len(gpus) > 0:
#     tf.config.experimental.set_memory_growth(gpus[0], True)

base_directory = '/home/ntustison/Data/WMH/'
scripts_directory = base_directory + 'Scripts/'

from batch_octant_generator import batch_generator

template = ants.image_read( antspynet.get_antsxnet_data("croppedMni152"))
patch_size = (112, 112, 112)

################################################
#
#  Create the model and load weights
#
################################################

classes = (0, 1)
number_of_classification_labels = len(classes)
image_modalities = ("T1", "FLAIR")
channel_size = len(image_modalities)

unet_model = antspynet.create_unet_model_3d((*patch_size, channel_size),
    number_of_outputs=number_of_classification_labels, mode="classification",
    number_of_layers=4, number_of_filters_at_base_layer=16, dropout_rate=0.0,
    convolution_kernel_size=(3, 3, 3), deconvolution_kernel_size=(2, 2, 2),
    weight_decay=1e-5, nn_unet_activation_style=False, add_attention_gating=True)

brain_weights_filename = scripts_directory + "wmhSegmentationWeights.h5"
if os.path.exists(brain_weights_filename):
    unet_model.load_weights(brain_weights_filename)

dice_loss = antspynet.multilabel_dice_coefficient(dimensionality=3)
surface_loss = antspynet.multilabel_surface_loss(dimensionality=3)

def combined_loss(alpha):
    def combined_loss_fixed(y_true, y_pred):
        return (alpha * dice_loss(y_true, y_pred) + 
                (1 - alpha) * surface_loss(y_true, y_pred))
    return(combined_loss_fixed)

wmh_loss = combined_loss(0.5)

unet_model.compile(optimizer=keras.optimizers.Adam(),
                   loss=wmh_loss,
                   metrics=['accuracy'])


################################################
#
#  Load the brain data
#
################################################

print("Loading brain data.")

t1_adni_images = glob.glob(base_directory + "ADNI_temp_mni_processed/*/*/*/*/*mnixT1_Preprocessed.nii.gz")
t1_oasis3_images = glob.glob(base_directory + "MniProcessed/*/*/NIFTI/*mnixT1_Preprocessed.nii.gz")

t1_images = (*t1_adni_images, *t1_oasis3_images)

training_t1_files = list()
training_flair_files = list()
training_seg_files = list()
training_mask_files = list()

for i in range(len(t1_images)):
    t1 = t1_images[i]
    flair = t1.replace("mnixT1_Preprocessed", "mnixT1xFlair")
    wmh = t1.replace("mnixT1_Preprocessed", "mnixT1_WMH")
    mask = t1.replace("mnixT1_Preprocessed", "mnixT1_BrainMask")

    if not os.path.exists(flair) or not os.path.exists(wmh) or not os.path.exists(mask):
        continue

    training_t1_files.append(t1)
    training_flair_files.append(flair)
    training_seg_files.append(wmh)
    training_mask_files.append(mask)

print("Total training image files: ", len(training_t1_files))

print( "Training")

###
#
# Set up the training generator
#

batch_size = 12 

# Split trainingData into "training" and "validation" componets for
# training the model.

number_of_data = len(training_t1_files)
sample_indices = random.sample(range(number_of_data), number_of_data)

validation_split = math.floor(0.8 * number_of_data)

training_indices = sample_indices[:validation_split]
number_of_training_data = len(training_indices)

sampled_training_t1_files = list()
sampled_training_flair_files = list()
sampled_training_mask_files = list()
sampled_training_seg_files = list()

for i in range(number_of_training_data):
    sampled_training_t1_files.append(training_t1_files[training_indices[i]])
    sampled_training_flair_files.append(training_flair_files[training_indices[i]])
    sampled_training_mask_files.append(training_mask_files[training_indices[i]])
    sampled_training_seg_files.append(training_seg_files[training_indices[i]])

validation_indices = sample_indices[validation_split:]
number_of_validation_data = len(validation_indices)

sampled_validation_t1_files = list()
sampled_validation_flair_files = list()
sampled_validation_mask_files = list()
sampled_validation_seg_files = list()

for i in range(number_of_validation_data):
    sampled_validation_t1_files.append(training_t1_files[validation_indices[i]])
    sampled_validation_flair_files.append(training_flair_files[validation_indices[i]])
    sampled_validation_mask_files.append(training_mask_files[validation_indices[i]])
    sampled_validation_seg_files.append(training_seg_files[validation_indices[i]])


track = unet_model.fit_generator(
   generator=batch_generator(batch_size=batch_size,
                             patch_size=patch_size,
                             template=template,
                             t1s=sampled_training_t1_files,
                             flairs=sampled_training_flair_files,
                             brain_masks=sampled_training_mask_files,
                             segmentation_images=sampled_training_seg_files,
                             segmentation_labels=classes,
                             do_random_contralateral_flips=False,
                             do_data_augmentation=False,
                            ),
    steps_per_epoch=48,
    epochs=200,
    validation_data=batch_generator(batch_size=batch_size,
                                    patch_size=patch_size,
                                    template=template,
                                    t1s=sampled_validation_t1_files,
                                    flairs=sampled_validation_flair_files,
                                    brain_masks=sampled_validation_mask_files,
                                    segmentation_images=sampled_validation_seg_files,
                                    segmentation_labels=classes,
                                    do_random_contralateral_flips=False,
                                    do_data_augmentation=False,
                                   ),
    validation_steps=12,
    callbacks=[
        keras.callbacks.ModelCheckpoint(brain_weights_filename, monitor='val_loss',
            save_best_only=True, save_weights_only=True, mode='auto', verbose=1),
        keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.5,
           verbose=1, patience=10, mode='auto'),
        keras.callbacks.EarlyStopping(monitor='val_loss', min_delta=0.001,
           patience=20, verbose=1)
        ]
    )

unet_model.save_weights(brain_weights_filename)



