import numpy as np
import random
import ants
import antspynet

def batch_generator(batch_size=32,
                    patch_size=(64, 64, 64),
                    template=None,
                    t1s=None,
                    flairs=None,
                    brain_masks=None,
                    segmentation_images=None,
                    segmentation_labels=None,
                    do_random_contralateral_flips=True,
                    do_data_augmentation=True):

   if template is None:
       raise ValueError("No reference template specified.")

   if t1s is None or flairs is None:
       raise ValueError("Input images must be specified.")

   if segmentation_images is None:
       raise ValueError("Input masks must be specified.")

   if segmentation_labels is None:
       raise ValueError("segmentation labels must be specified.")

   stride_length = tuple(np.subtract(template.shape, patch_size))

   while True:

       X = np.zeros((batch_size, *patch_size, 2))
       Y = np.zeros((batch_size, *patch_size))

       batch_count = 0

       while batch_count < batch_size:
           i = random.sample(range(len(t1s)), 1)[0]

           mask = ants.image_read(brain_masks[i])
           batch_wmh = ants.image_read(segmentation_images[i])

           batch_t1 = ants.image_read(t1s[i]) * mask
           batch_flair = ants.image_read(flairs[i]) * mask

           if do_random_contralateral_flips and random.sample((True, False), 1)[0]:
               t1A = t1.numpy()
               t1 = ants.from_numpy(t1A[t1A.shape[0]::0,:,:])
               flairA = flair.numpy()
               flair = ants.from_numpy(flairA[flairA.shape[0]::0,:,:])
               wmhA = batch_wmh.numpy()
               wmh = ants.from_numpy(wmhA[wmhA.shape[0]::0,:,:])
           else:
               t1 = batch_t1
               flair = batch_flair
               wmh = batch_wmh

           warped_t1 = t1
           warped_flair = flair
           warped_wmh = wmh

           wmh_patches = None
           t1_patches = None
           flair_patches = None

           if do_data_augmentation == True:
               data_augmentation = antspynet.randomly_transform_image_data(template,
                   [[warped_t1, warped_flair]],
                   [warped_wmh],
                   number_of_simulations=1,
                   transform_type='affineAndDeformation',
                   sd_affine=0.01,
                   deformation_transform_type="bspline",
                   number_of_random_points=1000,
                   sd_noise=2.0,
                   number_of_fitting_levels=4,
                   mesh_size=1,
                   sd_smoothing=4.0,
                   input_image_interpolator='linear',
                   segmentation_image_interpolator='nearestNeighbor')

               simulated_t1 = data_augmentation['simulated_images'][0][0]
               simulated_t1 = (simulated_t1 - simulated_t1.mean()) / simulated_t1.std()
               simulated_flair = data_augmentation['simulated_images'][0][1]
               simulated_flair = (simulated_flair - simulated_flair.mean()) / simulated_flair.std()

               simulated_wmh = data_augmentation['segmentation_images'][0]

               t1_patches = antspynet.extract_image_patches(simulated_t1, patch_size, max_number_of_patches='all',
                   stride_length=stride_length, random_seed=None, return_as_array=True)
               flair_patches = antspynet.extract_image_patches(simulated_flair, patch_size, max_number_of_patches='all',
                   stride_length=stride_length, random_seed=None, return_as_array=True)
               wmh_patches = antspynet.extract_image_patches(simulated_wmh, patch_size, max_number_of_patches='all',
                   stride_length=stride_length, random_seed=None, return_as_array=True)

           else:

               warped_t1 = (warped_t1 - warped_t1.mean()) / warped_t1.std()
               warped_flair = (warped_flair - warped_flair.mean()) / warped_flair.std()

               t1_patches = antspynet.extract_image_patches(warped_t1, patch_size, max_number_of_patches='all',
                   stride_length=stride_length, random_seed=None, return_as_array=True)
               flair_patches = antspynet.extract_image_patches(warped_flair, patch_size, max_number_of_patches='all',
                   stride_length=stride_length, random_seed=None, return_as_array=True)
               wmh_patches = antspynet.extract_image_patches(warped_wmh, patch_size, max_number_of_patches='all',
                   stride_length=stride_length, random_seed=None, return_as_array=True)

           which_octant = random.sample(range(8), 1)[0]

           if wmh_patches[which_octant,:,:,:].sum() >= 1000:
               X[batch_count,:,:,:,0] = t1_patches[which_octant,:,:,:]
               X[batch_count,:,:,:,1] = flair_patches[which_octant,:,:,:]
               Y[batch_count,:,:,:] = wmh_patches[which_octant,:,:,:]

               batch_count = batch_count + 1

       encoded_Y = antspynet.encode_unet(Y, segmentation_labels)

       yield X, encoded_Y, [None]









