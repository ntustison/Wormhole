import numpy as np
import random
import ants
import antspynet

def batch_generator(batch_size=32,
                    image_size=(64, 64),
                    t1s=None,
                    flairs=None,
                    segmentation_images=None,
                    segmentation_labels=None,
                    number_of_slices_per_image=5,
                    do_random_contralateral_flips=True,
                    do_histogram_intensity_warping=True,
                    do_add_noise=True,
                    do_data_augmentation=True):

    if t1s is None or flairs is None:
        raise ValueError("Input images must be specified.")

    if segmentation_images is None:
        raise ValueError("Input masks must be specified.")

    if segmentation_labels is None:
        raise ValueError("segmentation labels must be specified.")

    while True:

        X = np.zeros((batch_size, *image_size, 1))
        Y  = np.zeros((batch_size, *image_size))

        batch_count = 0

        while batch_count < batch_size:
            i = random.sample(range(len(segmentation_images)), 1)[0]

            flair = ants.image_read(flairs[i])
            t1 = ants.image_read(t1s[i])
            wmh = ants.image_read(segmentation_images[i])

            geoms = ants.label_geometry_measures(wmh)
            if len(geoms['Label']) == 0:
                continue

            resampling_params = np.array(ants.get_spacing(flair))

            do_resampling = False
            for d in range(len(resampling_params)):
                if resampling_params[d] < 0.8:
                    resampling_params[d] = 1.0
                    do_resampling = True

            resampling_params = tuple(resampling_params)

            if do_resampling:
                t1 = ants.resample_image(t1, resampling_params, use_voxels=False, interp_type=0)
                flair = ants.resample_image(flair, resampling_params, use_voxels=False, interp_type=0)
                wmh = ants.resample_image(wmh, resampling_params, use_voxels=False, interp_type=1)

            if do_random_contralateral_flips and random.sample((True, False), 1)[0]:
                t1A = t1.numpy()
                t1 = ants.from_numpy(t1A[t1A.shape[0]:-1:,:,:], origin=t1.origin,
                    spacing=t1.spacing, direction=t1.direction)
                flairA = flair.numpy()
                flair = ants.from_numpy(flairA[flairA.shape[0]:-1:,:,:], origin=flair.origin,
                    spacing=flair.spacing, direction=flair.direction)
                wmhA = wmh.numpy()
                wmh = ants.from_numpy(wmhA[wmhA.shape[0]:-1:,:,:], origin=wmh.origin,
                    spacing=wmh.spacing, direction=wmh.direction)

            if do_histogram_intensity_warping and random.sample((True, False), 1)[0]:
                break_points = [0.2, 0.4, 0.6, 0.8]
                displacements = list()
                for b in range(len(break_points)):
                    displacements.append(abs(random.gauss(0, 0.175)))
                    if random.sample((True, False), 1)[0]:
                        displacements[b] *= -1
                t1 = antspynet.histogram_warp_image_intensities(t1,
                    break_points=break_points, clamp_end_points=(True, False),
                    displacements=displacements)
                flair = antspynet.histogram_warp_image_intensities(flair,
                    break_points=break_points, clamp_end_points=(True, False),
                    displacements=displacements)


            if do_data_augmentation == True:
                data_augmentation = antspynet.randomly_transform_image_data(t1,
                    [[t1, flair]],
                    [wmh],
                    number_of_simulations=1,
                    transform_type='affine',
                    sd_affine=0.01,
                    deformation_transform_type="bspline",
                    number_of_random_points=1000,
                    sd_noise=2.0,
                    number_of_fitting_levels=4,
                    mesh_size=1,
                    sd_smoothing=4.0,
                    input_image_interpolator='linear',
                    segmentation_image_interpolator='nearestNeighbor')

                t1 = data_augmentation['simulated_images'][0][0]
                flair = data_augmentation['simulated_images'][0][1]
                wmh = data_augmentation['segmentation_images'][0]

            t1 = (t1 - t1.mean()) / t1.std()
            flair = (flair - flair.mean()) / flair.std()
            if do_add_noise and random.sample((True, False), 1)[0]:
                noise_parameters = (0.0, random.uniform(0, 0.05))
                t1 = ants.add_noise_to_image(t1, noise_model="additivegaussian", noise_parameters=noise_parameters)
                flair = ants.add_noise_to_image(flair, noise_model="additivegaussian", noise_parameters=noise_parameters)
                t1 = (t1 - t1.mean()) / t1.std()
                flair = (flair - flair.mean()) / flair.std()

            wmh_array = wmh.numpy()
            t1_array = t1.numpy()
            flair_array = flair.numpy()

            which_dimension_max_spacing = resampling_params.index(max(resampling_params))
            if which_dimension_max_spacing == 0:
                lower_slice = geoms['BoundingBoxLower_x'][0]
                upper_slice = geoms['BoundingBoxUpper_x'][0]
            elif which_dimension_max_spacing == 1:
                lower_slice = geoms['BoundingBoxLower_y'][0]
                upper_slice = geoms['BoundingBoxUpper_y'][0]
            else:
                lower_slice = geoms['BoundingBoxLower_z'][0]
                upper_slice = geoms['BoundingBoxUpper_z'][0]
            if lower_slice >= upper_slice:
                continue

            number_of_samples = min(number_of_slices_per_image, upper_slice - lower_slice)
            if number_of_samples <= 0:
                continue

            which_random_slices = random.sample(list(range(lower_slice, upper_slice)), number_of_samples)

            for j in range(len(which_random_slices)):
                which_slice = which_random_slices[j]

                t1_slice = None
                flair_slice = None
                wmh_slice = None

                if which_dimension_max_spacing == 0:
                    t1_slice = ants.from_numpy(np.squeeze(t1_array[which_slice,:,:]))
                    flair_slice = ants.from_numpy(np.squeeze(flair_array[which_slice,:,:]))
                    wmh_slice = ants.from_numpy(np.squeeze(wmh_array[which_slice,:,:]))
                elif which_dimension_max_spacing == 1:
                    t1_slice = ants.from_numpy(np.squeeze(t1_array[:,which_slice,:]))
                    flair_slice = ants.from_numpy(np.squeeze(flair_array[:,which_slice,:]))
                    wmh_slice = ants.from_numpy(np.squeeze(wmh_array[:,which_slice,:]))
                else:
                    t1_slice = ants.from_numpy(np.squeeze(t1_array[:,:,which_slice]))
                    flair_slice = ants.from_numpy(np.squeeze(flair_array[:,:,which_slice]))
                    wmh_slice = ants.from_numpy(np.squeeze(wmh_array[:,:,which_slice]))

                # Ensure that only ~10% are empty of any wmhs (i.e., voxel count < 100)
                if wmh_slice.sum() < 100 and random.uniform(0, 1) > 0.01:
                    continue

                t1_slice = antspynet.pad_or_crop_image_to_size(t1_slice, image_size)
                flair_slice = antspynet.pad_or_crop_image_to_size(flair_slice, image_size)
                wmh_slice = antspynet.pad_or_crop_image_to_size(wmh_slice, image_size)

                # X[batch_count,:,:,0] = flair_slice.numpy()
                X[batch_count,:,:,0] = t1_slice.numpy()
                Y[batch_count,:,:] = wmh_slice.numpy()

                batch_count = batch_count + 1
                if batch_count >= batch_size:
                    break

        encoded_Y = antspynet.encode_unet(Y, segmentation_labels)

        yield X, encoded_Y, [None]









