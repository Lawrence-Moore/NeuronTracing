import numpy as np
from skimage.transform import (hough_line, hough_line_peaks,
                               probabilistic_hough_line)
from skimage.feature import canny
import matplotlib.pyplot as plt
from sklearn import svm
from skimage.color import rgb2gray
from math import sqrt
from scipy import stats
import pickle


def learn_color_quantization(image, color_map):
    '''
    Learns to automatically separate a color space into the important areas
    color_map is a dictionary of color values as keys and a boolean as a value indicating if it's valid
    '''
    svc = svm.SVC(kernel='rbf')
    # split up the color map into training and testing examples
    np.random.seed(0)

    color_pixels = color_map.keys()
    color_status = color_map.values() 
    indices = np.random.permutation(len(color_pixels))

    # go through and generate features
    features = generate_data(color_pixels, color_status, image)

    num_train = .80 * len(color_pixels)

    x_train = features[indices[:num_train]]
    y_train = color_status[indices[:num_train]]

    x_test = features[indices[num_train:]]
    y_test = color_status[indices[num_train:]]
    svc.fit(x_train, y_train)


def generate_data(color_pixels, color_status, image):
    data = []
    radius = 15  # number of pixels
    for color, status in zip(color_pixels, color_status):
        color_features = []
        color_features.append(get_percentage_of_pixels_within_color_radius_feature(image, color, radius))
        color_features.append(get_length_of_lines_feature(image, color, radius))
        color_features.append(measure_increase_in_connectivity_with_radius_feature(image, color, radius))

        datum = [color_features, status]
        data.append(datum)

    # save to text
    pickle.dump(data, open("data.txt", 'wb'))
    return data


def generate_image_based_on_values(color, image, radius):
    red_mask = np.logical_and(image[:, :, 0] > color[0] - radius, image[:, :, 0] < color[0] + radius)
    green_mask = np.logical_and(image[:, :, 1] > color[1] - radius, image[:, :, 1] < color[1] + radius)
    blue_mask = np.logical_and(image[:, :, 2] > color[2] - radius, image[:, :, 2] < color[2] + radius)
    mask = np.logical_and.reduce([red_mask, green_mask, blue_mask])

    masked_image = image.copy()
    masked_image[np.logical_not(mask)] = 0
    return masked_image


def get_percentage_of_pixels_within_color_radius_feature(image, color, radius):
    # calculates how many pixels have the color in the speicified radius and how many don't
    return np.sum(image != 0) / np.sum(image == 0)


def measure_increase_in_connectivity_with_radius_feature(image, color, starting_radius, increment=200):
    # increment radius and see if it helps connectivity
    lengths = []
    radii = []
    for multiplyer in range(10):
        radius = starting_radius + multiplyer * increment
        radii.append(radius)
        lengths.append(get_length_of_lines_feature(image, color, radius))
    return stats.linregress(radii, lengths)[0]


def get_length_of_lines_feature(image, color, radius):
    edges = canny(image)
    lines = probabilistic_hough_line(edges, threshold=10, line_length=10, line_gap=3)
    return calculate_length_of_lines(lines)


def calculate_length_of_lines(lines):
    if len(lines) == 0:
        return 0
    else:
        line_length_sum = 0
        for line in lines:
            p0, p1 = line
            line_length_sum += sqrt((p0[0] - p1[0])**2 + (p0[1] - p1[1])**2)
        return float(line_length_sum) / len(lines)
