import numpy as np
from czifile import CziFile
from PIL import Image
import scipy
from matplotlib import pyplot as plt
from skimage.feature import match_template
from sklearn.cluster import KMeans
from sklearn.metrics import pairwise_distances_argmin
from sklearn.utils import shuffle


def normalize_with_min_max(images, threshold_min, threshold_max):
    return normalize_colors(images, thresholded_min_max=True, threshold_min=threshold_min, threshold_max=threshold_max)


def normalize_with_standard_deviation(images, std_multiple):
    return normalize_colors(images, threshold_std=True, std_multiple=std_multiple)


def normalize_generic(images):
    return normalize_colors(images)


def normalize_colors(images, thresholded_min_max=False, threshold_std=False, threshold_min=0, threshold_max=0, std_multiple=0):
    """
    Normalize the colors by standarding the mean and standard deviation throughout each layer
    Takes in a list of 3D numpy arrays representing the images. Returns a list of the same format with the images normalized
    """
    # find the overal mean and standard deviation
    if thresholded_min_max or threshold_std:
        # go through and threhold the images
        layer_means = []
        layer_stds = []
        for image in images:
            # get the mask
            if thresholded_min_max:
                mask = threshold_with_min_max(image, threshold_min, threshold_max)
            else:
                mask = threshold_with_standard_deviation(image, std_multiple)

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
    for index, image in enumerate(images):
        # iterate through the color layers
        layers = []
        for layer_index in [0, 1, 2]:
            layer = image[:, :, layer_index]

            if thresholded_min_max or threshold_std:
                if thresholded_min_max:
                    mask = threshold_with_min_max(layer, threshold_min, threshold_max)
                else:
                    mask = threshold_with_standard_deviation(layer, std_multiple)

                layer[mask] = (((layer[mask] - np.mean(layer[mask])) / np.std(layer[mask])) * std) + mean
                layers.append(layer)
            else:
                # first divide by mean and std_deviation, than scale by the overal mean and std from above
                layer = (((layer - np.mean(layer)) / np.std(layer)) * std) + mean
                layers.append(layer)

        normalized_layers.append(layers)

    # go through and cut any values less than 0 or greater than 2**16
    normalized_images = []
    for image in normalized_layers:
        image = np.stack(image, axis=2)
        image[image < 0] = 0
        image[image >= 2**16] = 2**16 - 1
        normalized_images.append(image.astype(np.uint16))
    return normalized_images


def threshold_with_standard_deviation(layer, std_multiple):
    '''
    Takes in a numpy array of an layer and returns a mask.
    The mask is true when the pixel values are within 3 standard deviations of the mean
    '''
    mean = np.mean(layer)
    std = np.std(layer)
    return np.logical_and(mean < layer, layer < mean + std_multiple * std)


def threshold_with_min_max(image, threshold_min, threshold_max):
    '''
    Takes in a numpy array of an image and returns a mask.
    The mask is true where the pixel values exceed the minimum
    '''
    return np.logical_and(threshold_min < image, image < threshold_max)


def generate_mip(images):
    '''
    Takes in a list of 3D numpy arrays representing images and returns a single 3D numpy array representing the MIP
    '''
    return np.maximum.reduce(images)


def save_images(images, name):
    '''
    Takes in a list of 3D numpy arrays representing images and saves each one to a tiff file individually
    '''
    for index, image in enumerate(images):
        file_name = name + " - z-layer " + index + ".tif"
        save_image(image, file_name)


def save_image(image, file_name):
    '''
    Takes in a single 3D numpy array representing an image and a string with the file name (including extension).
    Saves the image to the specified name
    '''
    import scipy.misc
    scipy.misc.toimage(image, cmin=0.0, cmax=2**16).save(file_name)


def align_images(images, manual=False, template_top_left_x=0,
                                       template_top_left_y=0,
                                       template_width=0,
                                       template_color_layer=0,
                                       template_image_index=0):
    # align blue and green and then blue and red
    if manual is True:
        patch_indexes = (template_top_left_x, template_top_left_y)
        patch = images[template_image_index][:, :, template_color_layer][template_top_left_x: template_top_left_x + template_width,
                                                                         template_top_left_y: template_top_left_y + template_width]
        # shift over the necessary color indexes
        color_indexes = [0, 1, 2]
        color_indexes.remove(template_color_layer)

        aligned_images = adjust_color_layer(images, color_indexes[0], patch_indexes, patch, template_image_index, template_width)
        aligned_images = adjust_color_layer(aligned_images, color_indexes[1], patch_indexes, patch, template_image_index, template_width)
        visualize_alignment(images, aligned_images, patch_indexes, template_image_index, template_width)
        return aligned_images
    else:
        # choose an image from them iddle
        image_index = len(images) / 2
        image = images[image_index]

        # find the brightest patch in any of the color layers
        width = 50
        patch_indexes, best_color, patch = get_bright_patch(image, width)

        color_indexes = [0, 1, 2]
        color_indexes.remove(best_color)

        aligned_images = adjust_color_layer(images, color_indexes[0], patch_indexes, patch, image_index, width)
        aligned_images = adjust_color_layer(aligned_images, color_indexes[1], patch_indexes, patch, image_index, width)
        visualize_alignment(images, aligned_images, patch_indexes, image_index, width)

        return aligned_images


def adjust_color_layer(images, color_index, patch_indexes, patch, image_index, width):
    '''
    Adjust the color layers so they align as closely as possible
    I'll use blue as the baseline truth for alignment
    '''

    # find that patch in blue and compare coordinates
    xy_offset, final_z_offset = match_patch(images, patch_indexes, patch, width, image_index, color_index)
    print xy_offset, final_z_offset

    # shift the layer according to the offset
    return shift_color_chanel((xy_offset[0], xy_offset[1], final_z_offset), images, color_index)


def get_bright_patch(image, width):
    '''
    Given a 3D image numpy array, the width of the patch, and color index corresponding to the index of the color layer,
    find a region with the highest standard deviation to use for template matching
    '''
    highest_minimum_std = 0
    coordinates = [0, 0]
    best_color = 0

    layer = image[:, :, 0]
    # iterate through, finding the region with the highest standard devation
    for i in range(image.shape[0] / 4, 3 * image.shape[0] / 4)[::width]:
        for j in range(image.shape[1] / 4, 3 * image.shape[1] / 4)[::width]:
            stds = []
            # go through each color layer and calculate the std
            for color in [0, 1, 2]:
                layer = image[:, :, color]
                stds.append(np.std(layer[i: i + width, j: j + width]))

            if np.min(stds) > highest_minimum_std:
                highest_minimum_std = np.min(stds)
                coordinates = [i, j]
                best_color = np.argmin(stds)

    # get the patch
    patch = image[:, :, best_color][coordinates[0]: coordinates[0] + width, coordinates[1]: coordinates[1] + width]

    # subtract width to get the top left of the square
    return (coordinates, best_color, patch)


def match_patch(images, patch_index, patch, width, image_index, color_index, wiggle_room=20):
    result = 0
    final_z_offset = 0

    # go through and find the greatest similarity in the layers
    for z_offset in range(5):
        image = images[image_index + z_offset]

        # get the region to check. It should be the size of the patch plus some wiggle room around the edges
        comparison_area = image[patch_index[0] - wiggle_room: patch_index[0] + width + wiggle_room,
                                patch_index[1] - wiggle_room: patch_index[1] + width + wiggle_room, color_index]
        comparison_area = np.reshape(comparison_area, (wiggle_room * 2 + width, wiggle_room * 2 + width))

        # perform the template matching
        new_result = match_template(comparison_area, patch)
        if np.max(new_result) > np.max(result):
            result = new_result
            final_z_offset = z_offset

    xy = np.unravel_index(np.argmax(result), result.shape)
    return np.array(xy) - wiggle_room, final_z_offset


def shift_color_chanel(offset, images, color_chanel):
    """
    Shift the color channel in each image according the offset.

    offset is a tuple of three values in the following format: (x_offset, y_offset, z_offset)
    images is a list of 3D numpy arrays representing the image
    color_chanel is the index of the color chanel being adjusted
    """
    shifted_images = []
    for image in images:
        # shift the layer accordingly
        shifted_image = image.copy()
        layer = shifted_image[:, :, color_chanel]

        # lateral
        if offset[0] < 0:
            # go down
            layer = np.pad(layer, ((abs(offset[0]), 0), (0, 0)), mode='constant')[:offset[0], :]
        elif offset[0] > 0:
            # go up
            layer = np.pad(layer, ((0, abs(offset[0])), (0, 0)), mode='constant')[abs(offset[0]):, :]

        # horizantal
        if offset[1] < 0:
            # shift right
            layer = np.pad(layer, ((0, 0), (abs(offset[1]), 0)), mode='constant')[:, :offset[1]]
        elif offset[1] > 0:
            # shift left
            layer = np.pad(layer, ((0, 0), (0, abs(offset[1]))), mode='constant')[:, abs(offset[1]):]

        shifted_image[:, :, color_chanel] = layer
        shifted_images.append(shifted_image)

    # now shift vertically
    if offset[2] > 0:
        # shift the z layer of the chanel up
        for i in range(len(images) - abs(offset[2])):
            images[i][:, :, color_chanel] = images[i + abs(offset[2])][:, :, color_chanel]

        # fill in the rest with zeros
        for i in range(len(images) - abs(offset[2]), len(images)):
            images[i][:, :, color_chanel] = np.zeros(images[i][:, :, color_chanel].shape)

    elif offset[2] < 0:
        # shift the z layer of the chanel down
        for i in reversed(range(abs(offset[2]), len(images))):
            images[i][:, :, color_chanel] = images[i - abs(offset[2])][:, :, color_chanel]

        for i in range(abs(offset[2])):
            images[i][:, :, color_chanel] = np.zeros(images[i][:, :, color_chanel].shape)

    return shifted_images


def visualize_alignment(images, aligned_images, patch_indexes, image_index, width):
    '''
    Compare the patch before and after in the images
    '''
    old_patch = images[image_index][patch_indexes[0]: patch_indexes[0] + width, patch_indexes[1]: patch_indexes[1] + width, :]
    new_patch = aligned_images[image_index][patch_indexes[0]: patch_indexes[0] + width, patch_indexes[1]: patch_indexes[1] + width, :]
    return old_patch, new_patch


def evaluate_normalization_with_mask(old_images, adjusted_images, mask):
    """
    Evaluates the correction by plotting the mean of the area specified by the mask
    Takes in a list of 3D numpy arrays before and after the adjustments
    """
    # go through each layer and calculate the mean of the intensity of the pixels in interest
    old_green_means = []
    old_red_means = []
    old_blue_means = []
    new_green_means = []
    new_red_means = []
    new_blue_means = []
    for old_image, new_image in zip(old_images, adjusted_images):
        # calculate the means
        old_blue_means.append(np.mean(old_image[:, :, 0][mask]))
        old_green_means.append(np.mean(old_image[:, :, 1][mask]))
        old_red_means.append(np.mean(old_image[:, :, 2][mask]))

        # calculate the stds
        new_blue_means.append(np.mean(new_image[:, :, 0][mask]))
        new_green_means.append(np.mean(new_image[:, :, 1][mask]))
        new_red_means.append(np.mean(new_image[:, :, 2][mask]))

    # plot the stats
    plt.plot(old_blue_means, 'bo')
    plt.plot(old_green_means, 'go')
    plt.plot(old_red_means, 'ro')
    plt.plot(new_blue_means, 'b^')
    plt.plot(new_green_means, 'g^')
    plt.plot(new_red_means, 'r^')

    plt.xlabel('Depth in millimeters')
    plt.ylabel('Mean Color Intensity in the region')
    plt.title('Color Attenuation Correction')

    plt.legend(['old blue means', 'old green_means', 'old red means', 'new blue stds', 'new green stds', 'new red stds'])
    plt.show()


def visualize_mask(image, mask, name):
    '''
    Given a 2D numpy array (image) and a mask with the same size, save the masked image to the file name
    '''
    img = image.copy()
    img[np.logical_not(mask)] = 0
    scipy.misc.imsave(name + '.jpg', img)


def k_means(image, n_colors=64):
    # works better with values between 0 and 1
    image = np.array(image, dtype=np.float64) / (2**16 - 1)

    # reshape so it's 2-D
    w, h, d = original_shape = tuple(image.shape)
    image_array = np.reshape(image, (w * h, d))

    # fit on sample
    image_array_sample = shuffle(image_array, random_state=0)[:1000]
    kmeans = KMeans(n_clusters=n_colors, random_state=0).fit(image_array_sample)

    # Get labels for all points
    labels = kmeans.predict(image_array)

    # randomely choose colors
    codebook_random = shuffle(image_array, random_state=0)[:n_colors + 1]
    labels_random = pairwise_distances_argmin(codebook_random, image_array, axis=0)

    def recreate_image(codebook, labels, w, h):
        """Recreate the (compressed) image from the code book & labels"""
        d = codebook.shape[1]
        image = np.zeros((w, h, d))
        label_idx = 0
        for i in range(w):
            for j in range(h):
                image[i][j] = codebook[labels[label_idx]]
                label_idx += 1
        return image
    plt.figure(1)
    plt.clf()
    ax = plt.axes([0, 0, 1, 1])
    plt.axis('off')
    plt.title('Original image (96,615 colors)')
    plt.imshow(image)

    plt.figure(2)
    plt.clf()
    ax = plt.axes([0, 0, 1, 1])
    plt.axis('off')
    plt.title('Quantized image (64 colors, K-Means)')
    plt.imshow(recreate_image(kmeans.cluster_centers_, labels, w, h))

    plt.figure(3)
    plt.clf()
    ax = plt.axes([0, 0, 1, 1])
    plt.axis('off')
    plt.title('Quantized image (64 colors, Random)')
    plt.imshow(recreate_image(codebook_random, labels_random, w, h))
    plt.show()


def self_organization_map(image, n_colors=64):
    pass


def read_czi_file(file_name):
    """
    Reads in a czi file (the file name should include the extension) and returns a list of 3D numpy arrays representing the images
    Based on http://schryer.github.io/python_course_material/python/python_10.html
    """
    with CziFile(file_name) as czi:
        images = czi.asarray()

    # unpacks the values from the czi format.
    images = [np.rollaxis(images[0, :, 0, index, :, :, 0], 0, 3) for index in range(images.shape[3])]
    return images


def read_tiff_image(file_name):
    """
    Takes in a string of the filename including extension and returns the image
    """
    return np.array(Image.open(file_name))


def separate_colors(image, colors, image_name):
    '''
    Given an image (3D numpy array) and list of 3 element arrays representing colors, split it up into several different colors
    '''
    for index, color in enumerate(colors):
        img = image.copy()
        img[img != color] = 0
        save_image(img, image_name + " color " + str(index) + ".jpg")


def normalize_and_align():
    images = read_czi_file("../../880 BI/TBX DIO TRE-XFP ScalesSQ 20x 1.0Wtile stack_Subset_Stitch.czi")
    images = read_czi_file("../../880 BI/OB stack sparse low dose TBX 10x 70um.czi")
    return images
