import numpy as np
from czifile import CziFile
from PIL import Image
import scipy
from matplotlib import pyplot as plt
from skimage.feature import match_template
from sklearn.cluster import KMeans
from sklearn.utils import shuffle
from skimage.measure import structural_similarity as ssim
from math import sqrt
from minisom import MiniSom


def normalize_with_standard_deviation(images, std_multiple):
    return normalize_colors(images, threshold_std=True, std_multiple=std_multiple)


def normalize_generic(images):
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


def align_images(images, wiggle_room=20, manual=False, template_top_left_x=0,
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

        aligned_images, offsets1 = adjust_color_layer(images, color_indexes[0], patch_indexes, patch, template_image_index, template_width, wiggle_room=wiggle_room)
        aligned_images, offsets2 = adjust_color_layer(aligned_images, color_indexes[1], patch_indexes, patch, template_image_index, template_width, wiggle_room=wiggle_room)
        if template_image_index >= len(aligned_images):
            template_image_index = len(aligned_images) - 1
        visualize_alignment(images, aligned_images, patch_indexes, template_image_index, template_width, color_indexes, offsets1, offsets2)
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

        aligned_images, offsets1 = adjust_color_layer(images, color_indexes[0], patch_indexes, patch, image_index, width)
        aligned_images, offsets2 = adjust_color_layer(aligned_images, color_indexes[1], patch_indexes, patch, image_index, width)
        if template_image_index >= len(aligned_images):
            template_image_index = len(aligned_images) - 1
        visualize_alignment(images, aligned_images, patch_indexes, image_index, width, color_indexes, offsets1, offsets2)

        return aligned_images


def adjust_color_layer(images, color_index, patch_indexes, patch, image_index, width, wiggle_room=20):
    '''
    Adjust the color layers so they align as closely as possible
    I'll use blue as the baseline truth for alignment
    '''

    # find that patch in blue and compare coordinates
    offsets, best_patch = match_patch(images, patch_indexes, patch, width, image_index, color_index, wiggle_room=wiggle_room)
    # print "one way", offsets
    # offsets, best_patch = alternative_patch_matching(images, patch_indexes, patch, width, image_index, color_index)
    # print "another", offsets

    # visualize the match
    print patch.shape, best_patch.shape
    fig = plt.figure()
    ax = fig.add_subplot(1, 2, 1)
    plt.imshow((patch.astype(np.float16) / (2 ** 16-1) * (2 ** 8-1)).astype(np.uint8))
    ax.set_title('original patch')
    ax = fig.add_subplot(1, 2, 2)
    plt.imshow((best_patch.astype(np.float16) / (2 ** 16 - 1) * (2 ** 8 - 1)).astype(np.uint8))
    ax.set_title('matched patch')

    # shift the layer according to the offset
    return shift_color_chanel(offsets, images, color_index), offsets


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
    # check for the range
    offset = 20
    min_shift = -1 * min(offset, image_index)
    max_shift = min(offset + image_index, len(images)) - image_index
    for z_offset in range(min_shift, max_shift):
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

    # adjust the coordinates
    xy_offset = np.array(xy) - wiggle_room
    best_patch = images[image_index + final_z_offset][patch_index[0] + xy_offset[0]: patch_index[0] + xy_offset[0] + width,
                                                      patch_index[1] + xy_offset[1]: patch_index[1] + xy_offset[1] + width, color_index]

    return (xy_offset[0], xy_offset[1], final_z_offset), best_patch


def alternative_patch_matching(images, patch_index, patch, width, image_index, color_index, wiggle_room=5):
    '''
    Uses image comparsion instead of fuzzy matching
    '''
    best_result = 0
    final_z_offset = 0   
    for z_offset in range(-20, 20):
        image = images[image_index + z_offset]
        # iterate through squares and check if the images are similar
        for i in range(patch_index[0] - wiggle_room, patch_index[0] + wiggle_room):
            for j in range(patch_index[1] - wiggle_room, patch_index[1] + wiggle_room):
                comparison_area = image[i: i + width, j: j + width, color_index]
                comparison_score = ssim(patch, comparison_area)
                if comparison_score > best_result:
                    best_result = comparison_score
                    final_z_offset = z_offset
                    best_xy = [i, j]

    xy_offset = np.array(best_xy) - patch_index
    best_patch = images[image_index + final_z_offset][patch_index[0] + xy_offset[0]: patch_index[0] + xy_offset[0] + width,
                                                      patch_index[1] + xy_offset[1]: patch_index[1] + xy_offset[1] + width, color_index]

    return (xy_offset[0], xy_offset[1], final_z_offset), best_patch


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
        for i in range(len(shifted_images) - abs(offset[2])):
            shifted_images[i][:, :, color_chanel] = shifted_images[i + abs(offset[2])][:, :, color_chanel]

        # fill in the rest with zeros
        # for i in range(len(shifted_images) - abs(offset[2]), len(shifted_images)):
        #     shifted_images[i][:, :, color_chanel] = np.zeros(shifted_images[i][:, :, color_chanel].shape)

        shifted_images = shifted_images[:len(shifted_images) - abs(offset[2])]

    elif offset[2] < 0:
        # shift the z layer of the chanel down
        for i in reversed(range(abs(offset[2]), len(shifted_images))):
            shifted_images[i][:, :, color_chanel] = shifted_images[i - abs(offset[2])][:, :, color_chanel]

        # for i in range(abs(offset[2])):
        #     shifted_images[i][:, :, color_chanel] = np.zeros(shifted_images[i][:, :, color_chanel].shape)
        shifted_images = shifted_images[abs(offset[2]):]

    return shifted_images


def visualize_alignment(images, aligned_images, patch_indexes, image_index, width, color_indexes, offsets1, offsets2):
    '''
    Compare the patch before and after in the images
    '''
    old_patch = images[image_index][patch_indexes[0]: patch_indexes[0] + width, patch_indexes[1]: patch_indexes[1] + width, :]
    new_patch = aligned_images[image_index][patch_indexes[0]: patch_indexes[0] + width, patch_indexes[1]: patch_indexes[1] + width, :]

    fig = plt.figure()
    ax = fig.add_subplot(1, 2, 1)
    plt.imshow((old_patch.astype(np.float16) / (2 ** 16-1) * (2 ** 8-1)).astype(np.uint8))
    ax.set_title('Before')
    ax = fig.add_subplot(1, 2, 2)
    plt.imshow((new_patch.astype(np.float16) / (2 ** 16 - 1) * (2 ** 8 - 1)).astype(np.uint8))
    ax.set_title('After')

    color_layer1_adjust = 'For color layer %d, x offset: %d, y offset: %d, z offset: %d' % (color_indexes[0], offsets1[0], offsets1[1], offsets1[2])
    color_layer2_adjust = 'For color layer %d, x offset: %d, y offset: %d, z offset: %d' % (color_indexes[1], offsets2[0], offsets2[1], offsets2[2])
    adjustment_info = color_layer1_adjust + "\n" + color_layer2_adjust
    ax.text(-25, -10, adjustment_info, style='italic', bbox={'facecolor': 'red', 'alpha': 0.5, 'pad': 10})
    plt.show()

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


def visualize_mask(image, mask):
    '''
    Given a 2D numpy array (image) and a mask with the same size, save the masked image to the file name
    '''
    img = image.copy()
    img[np.logical_not(mask)] = 0
    display_image(img)


def display_image(image):
    plt.imshow((image.astype(np.float16) / (2 ** 16-1) * (2 ** 8-1)).astype(np.uint8))
    plt.show()


def k_means(image=None, images=None, weights=None, n_colors=64, num_training=1000, std_multiple=0, threshold=False, show_plot=False):
    neuron_pixels, non_neuron_pixels, image_array, image = sample_data(image, images, std_multiple)

    if threshold:
        image_array_sample = shuffle(neuron_pixels, random_state=0)[:num_training]
    else:
        image_array_sample = shuffle(image_array, random_state=0)[:num_training]

    if weights is not None:
        # reshape weights appropiately. Assumes a list of lists is passed in
        weights = [np.array(weight) for weight in weights]
        weights = np.vstack(tuple(weights))

        # make sure it's on the 0-1 scale
        weights = weights / (255) * (2**16 - 1)

        kmeans = KMeans(n_clusters=n_colors, n_init=1, init=weights).fit(image_array_sample)
    else:
        kmeans = KMeans(n_clusters=n_colors, random_state=0).fit(image_array_sample)

    if show_plot:
        # Get labels for all points
        labels = kmeans.predict(image_array)

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

        w, h, d = tuple(image.shape)
        quantized_image = recreate_image(kmeans.cluster_centers_, labels, w, h)

        # display the image
        fig = plt.figure()
        ax = fig.add_subplot(1, 2, 1)
        plt.imshow(image)
        ax.set_title('Original')
        ax = fig.add_subplot(1, 2, 2)
        plt.imshow(quantized_image)
        ax.set_title('After K-Means')
        plt.show()

    return kmeans.cluster_centers_


def self_organizing_map(image=None, images=None, weights=None, weights_max_value=1, n_colors=64,
                        dim=None, num_training=1000, std_multiple=0, threshold=False, show_plot=False):
    """
    Cluster an image using a self organizing map
    """
    neuron_pixels, non_neuron_pixels, pixels, image = sample_data(image, images, std_multiple)

    if dim is None and weights is not None:
        # figure out a way to spread out the nodes of the som
        # find the factor closest to the square root
            # normalize weights
        weights = (np.array(weights) / weights_max_value).tolist()

        factor = get_factor_closest_to_sqrt(len(weights))

        # it's prime if the factor is 1
        if factor == 1:
            # add a random weight to make it even
            weights = np.vstack((weights, np.random.random(3)))
        # should be fine now
        factor = get_factor_closest_to_sqrt(len(weights))
        dim = (factor, len(weights) / factor)
        weights = np.reshape(weights, (dim[0], dim[1], 3))

    else:
        if n_colors == 2 or n_colors == 3:
            dim = (1, n_colors)
        else:
            factor = get_factor_closest_to_sqrt(n_colors)
            # it's prime if the factor is 1
            if factor == 1:
                # increase the number of colors by one
                n_colors += 1
            # should be fine now
            factor = get_factor_closest_to_sqrt(n_colors)
            dim = (factor, n_colors / factor)

    # determine the dimensions
    som = MiniSom(dim[0], dim[1], 3, weights=weights, sigma=0.1, learning_rate=0.2)
    if weights is None:
        som.random_weights_init(pixels)

    if threshold:
        # get mostly bright pixels with a bit of background
        som.train_random(neuron_pixels, num_training)
    else:
        som.train_random(pixels, num_training)

    if show_plot:
        qnt = som.quantization(pixels)  # quantize each pixels of the image
        clustered = np.zeros(image.shape)
        for i, q in enumerate(qnt):
            clustered[np.unravel_index(i, dims=(image.shape[0], image.shape[1]))] = q

        fig = plt.figure()
        ax = fig.add_subplot(1, 2, 1)
        plt.imshow(image)
        ax.set_title('Original')
        ax = fig.add_subplot(1, 2, 2)
        plt.imshow(clustered)
        ax.set_title('After SOM Clustering')
        plt.show()

    return np.reshape(som.weights, (som.weights.shape[0] * som.weights.shape[1], 3))


def sample_data(image=None, images=None, std_multiple=0):
    if image is None and images is None:
        raise ValueError('No images were passed in!')
    if image is not None:
        # works better with values between 0 and 1
        image = np.array(image, dtype=np.float64) / (np.max(image))

        # reshape so it's 2-D
        w, h, d = tuple(image.shape)
        image_pixels = np.reshape(image, (w * h, d))

        # fit on sample
        mask = find_non_background_pixels(image, std_multiple)
        neuron_pixels, non_neuron_pixels = image[mask], image[np.logical_not(mask)]
    else:
        all_neuron_pixels = np.zeros((1, 3))
        sample_non_neuron_pixels = np.zeros((1, 3))
        # flatten out all the arrays into a single array
        for image in images:
            # works better with values between 0 and 1
            image = np.array(image, dtype=np.float64) / (np.max(image))
            mask = find_non_background_pixels(image, std_multiple)
            neuron_pixels, non_neuron_pixels = image[mask], image[np.logical_not(mask)]
            non_neuron_pixels = shuffle(non_neuron_pixels, random_state=0)[:int(.05 * neuron_pixels.shape[0])]  # keep only a small sample
            all_neuron_pixels = np.concatenate((all_neuron_pixels, neuron_pixels))
            sample_non_neuron_pixels = np.concatenate((sample_non_neuron_pixels, non_neuron_pixels))

        # just for consistencies sake, generate the mip
        image = generate_mip(images)
        image = np.array(image, dtype=np.float64) / (np.max(image))
        w, h, d = tuple(image.shape)
        image_pixels = np.reshape(image, (w * h, d))

    return neuron_pixels, non_neuron_pixels, image_pixels, image


def find_non_background_pixels(image, std_multiple=0):
    r_mask = image[:, :, 0] > np.mean(image[:, :, 0]) + std_multiple * np.std(image[:, :, 0])
    g_mask = image[:, :, 1] > np.mean(image[:, :, 1]) + std_multiple * np.std(image[:, :, 1])
    b_mask = image[:, :, 2] > np.mean(image[:, :, 2]) + std_multiple * np.std(image[:, :, 2])
    return np.logical_or.reduce([r_mask, g_mask, b_mask])


def get_factor_closest_to_sqrt(number):
    factor = int(sqrt(number))
    isFactor = number % factor == 0
    while not isFactor:
        factor -= 1
        isFactor = number % factor == 0
    return factor


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
