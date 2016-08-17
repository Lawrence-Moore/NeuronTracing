import numpy as np
from matplotlib import pyplot as plt
from minisom import MiniSom
from math import sqrt
from sklearn.utils import shuffle
from sklearn.cluster import KMeans
from helper_functions import generate_mip
from code.Masking.saving_and_color import xyv2rgb


def k_means(image=None, images=None, weights=None, n_colors=64, num_training=1000, std_multiple=0,
            threshold=False, show_plot=False, show_color_space_assignment=False):
    """
    Given an image or stack of images (list of np arrays), perform k means and return a list of the color centers
    num_training is the number of pixels used to train
    weights is a list of rgb values to initialize the algorithm with
    n_colors is the number of clusters
    threshold is a boolean whether to consider the black pixels or not
    std_multiple is the standard deviation multiple used in thresholding
    show_plot is a boolean whether to show the quantized image or not
    show_color_space_assignment is a boolean whether to show the way the color space is clustered or not
    """
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

    centers = kmeans.cluster_centers_

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

    if show_color_space_assignment:
        # use multiple intensities for the color space
        for intensity in [.1, .2, .4, .6, .8]:
            visualize_color_space(kmeans=kmeans, intensity=intensity)

    return centers


def self_organizing_map(image=None, images=None, weights=None, weights_max_value=1, n_colors=64,
                        dim=None, num_training=1000, std_multiple=0, threshold=False, show_plot=False, show_color_space_assignment=False):
    """
    Given an image or stack of images (list of np arrays), cluster with an SOM and return a list of the color centers
    num_training is the number of pixels used to train
    weights is a list of rgb values to initialize the algorithm with
    weights_max_value is a number representing the maximum possible value so that weights can be normalized to a 0-1 scale
    n_colors is the number of clusters
    dim is the dimensions of the nodes in the SOM. Total number of nodes should be the same as the number specified
    threshold is a boolean whether to consider the black pixels or not
    std_multiple is the standard deviation multiple used in thresholding
    show_plot is a boolean whether to show the quantized image or not
    show_color_space_assignment is a boolean whether to show the way the color space is clustered or not

    """
    neuron_pixels, non_neuron_pixels, pixels, image = sample_data(image, images, std_multiple)

    if dim is None and weights is not None:
        # normalize weights
        weights = (np.array(weights) / weights_max_value).tolist()

        # figure out a way to spread out the nodes of the som and find the int factor closest to the square root
        factor = get_factor_closest_to_sqrt(len(weights))

        # it's prime if the factor is 1
        if factor == 1:
            # add a random weight to make the number of nodes even
            weights = np.vstack((weights, np.random.random(3)))

        # should be fine now
        factor = get_factor_closest_to_sqrt(len(weights))
        dim = (factor, len(weights) / factor)
        weights = np.reshape(weights, (dim[0], dim[1], 3))

    else:
        # there are no weights to initialize
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
        if threshold:
            som.random_weights_init(neuron_pixels)
        else:
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

    if show_color_space_assignment:
        for intensity in [.1, .2, .4, .6, .8]:
            visualize_color_space(som=som, intensity=intensity)

    return np.reshape(som.weights, (som.weights.shape[0] * som.weights.shape[1], 3))


def sample_data(image=None, images=None, std_multiple=0):
    """
    Samples the image or (images) passed in based on the standard deviation multiple passed in.
    returns 4 objects: pixels belonging to neurons, pixels belonging to the background,
        all the pixels of the image, and a scaled version of the original image.
    The first three returns are numpy arrays of size [x, 3], where x is the number of pixels in each
    The final return type is the same size as the original image, just sacled to 0 - 1.
    """
    if image is None and images is None:
        raise ValueError('No images were passed in!')
    if image is not None:
        # works better with values between 0 and 1
        image = np.array(image.copy(), dtype=np.float64) / (np.max(image))

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

            # segregate the image based on the intensity
            mask = find_non_background_pixels(image, std_multiple)

            # split up into pixel and non pixel images
            neuron_pixels, non_neuron_pixels = image[mask], image[np.logical_not(mask)]

            # sample some background pixels
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
    """
    Gets the int factor closest to the square root
    """
    factor = int(sqrt(number))
    isFactor = number % factor == 0
    while not isFactor:
        factor -= 1
        isFactor = number % factor == 0
    return factor


def visualize_color_space(kmeans=None, som=None, intensity=.6, step = .5):
    """
    Allows the user to visualize the color space separation
    intensity is on a 0 to 1 scale
    step size of the mesh. Decrease to increase the quality of the visualization
    """

    # make the grid for HSV and plot the decision boundary. For that, we will assign a color to each
    radius = 100
    xx, yy = np.meshgrid(np.arange(0, radius * 2, step), np.arange(0, radius * 2, step))

    xyv = np.reshape(np.c_[xx.ravel(), yy.ravel(), np.ones(xx.shape).astype(np.float32).ravel() * intensity], (xx.shape[0], xx.shape[1], 3)).astype(np.float32)

    rgb_color_values = xyv2rgb(xyv, radius, 'hsv').astype(np.float32) / 255

    # Obtain labels for each point in mesh. Use last trained model.
    if kmeans:
        Z = kmeans.predict(np.reshape(rgb_color_values, (rgb_color_values.shape[0] * rgb_color_values.shape[1], 3)))
        print Z
    else:
        Z = som.predict(np.reshape(rgb_color_values, (rgb_color_values.shape[0] * rgb_color_values.shape[1], 3)))

    Z = Z.reshape(xx.shape)
    plt.figure(1)
    plt.clf()
    plt.imshow(Z, interpolation='nearest',
               extent=(xx.min(), xx.max(), yy.min(), yy.max()),
               cmap=plt.cm.Paired,
               aspect='auto', origin='lower')

    plt.xlim(0, radius * 2)
    plt.ylim(0, radius * 2)
    plt.xticks(())
    plt.yticks(())
    plt.show()
