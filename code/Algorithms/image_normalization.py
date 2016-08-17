import numpy as np


def normalize_with_standard_deviation(images, std_multiple):
    """
    Normalize using only the pixels greater than the mean and less than the mean plus std_multiple * standard devation
    """
    return normalize_colors(images, threshold_std=True, std_multiple=std_multiple)


def normalize_generic(images):
    """
    Normalize without threshold
    """
    return normalize_colors(images)


def normalize_colors(images, threshold_std=False, std_multiple=0):
    """
    Normalize the colors by standarding the mean and standard deviation throughout each layer
    Takes in a list of 3D numpy arrays representing the images. Returns a list of the same format with the images normalized
    """
    # find the overal mean and standard deviation
    if threshold_std:
        # go through and threhold the images
        layer_means = []
        layer_stds = []
        masks = []
        for image in images:
            # get the mask
            mask = threshold_with_standard_deviation(image, std_multiple)
            masks.append(mask)

            # apply the mask to image and get the stats
            thresholded_image = image[mask]
            layer_means.append(np.mean(thresholded_image))
            layer_stds.append(np.std(thresholded_image))
        # calculate the mean and stds of the thresholded layers by averaging both across the entire set of images
        mean = np.mean(layer_means)
        std = np.mean(layer_stds)
    else:
        mean = np.mean(images)
        std = np.std(images)

    # go through the images and calculate the mean and standard deviation of each
    normalized_layers = []
    if threshold_std:
        for image, mask in zip(images, masks):
            image = image.copy()
            image[mask] = (((image[mask] - np.mean(image[mask])) / np.std(image[mask])) * std) + mean
            normalized_layers.append(image)
    else:
        for image in images:
            # iterate through the color layers
            image = image.copy()
            layers = []
            for layer_index in [0, 1, 2]:
                layer = image[:, :, layer_index].copy()

                # first divide by mean and std_deviation, than scale by the overal mean and std from above
                layer = (((layer - np.mean(layer)) / np.std(layer)) * std) + mean
                layers.append(layer)

            normalized_layers.append(np.stack(layers, axis=2))

    # go through and cut any values less than 0 or greater than 2**16
    normalized_images = []
    for image in normalized_layers:
        image[image < 0] = 0
        image[image >= 2**16] = 2**16 - 1
        normalized_images.append(image.astype(np.uint16))
    return normalized_images


def threshold_with_standard_deviation(image, std_multiple):
    '''
    Takes in a numpy array of an layer and returns a mask.
    The mask is true when the pixel values are within 3 standard deviations of the mean
    '''
    red_mask = np.logical_and(np.mean(image[:, :, 0]) < image[:, :, 0], image[:, :, 0] < np.mean(image[:, :, 0]) + std_multiple * np.std(image[:, :, 0]))
    green_mask = np.logical_and(np.mean(image[:, :, 1]) < image[:, :, 1], image[:, :, 1] < np.mean(image[:, :, 1]) + std_multiple * np.std(image[:, :, 1]))
    blue_mask = np.logical_and(np.mean(image[:, :, 2]) < image[:, :, 2], image[:, :, 2] < np.mean(image[:, :, 2]) + std_multiple * np.std(image[:, :, 2]))
    return np.logical_or.reduce([red_mask, green_mask, blue_mask])


def threshold_with_min_max(image, threshold_min, threshold_max):
    '''
    Takes in a numpy array of an image and returns a mask.
    The mask is true where the pixel values exceed the minimum
    '''
    return np.logical_and(threshold_min < image, image < threshold_max)
