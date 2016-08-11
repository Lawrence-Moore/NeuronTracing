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

    # get the indexes and values of the color map
    pixel_colors = []
    pixel_status = []
    for intensity in range(color_map.shape[0]):
        for x in range(color_map.shape[1]):
            for y in range(color_map.shape[2]):
                pixel_colors.append((intensity, x, y))
                pixel_status.append(color_map[intensity, x, y])

    indices = np.random.permutation(len(pixel_colors))

    # go through and generate features
    features = generate_data(pixel_colors, pixel_status, image)

    num_train = .80 * len(features)

    x_train = features[indices[:num_train]]
    y_train = pixel_status[indices[:num_train]]

    x_test = features[indices[num_train:]]
    y_test = pixel_status[indices[num_train:]]
    svc.fit(x_train, y_train)
    print svc.score(x_test, y_test)


def generate_data(pixel_colors, pixel_status, image):
    feature_one = []
    feature_two = []
    feature_three = []
    radius = 15  # number of pixels
    for color, status in zip(pixel_colors, pixel_status):
        color_features = []
        masked_image = generate_image_based_on_values(image, color, radius)
        feature_one.append(get_percentage_of_pixels_within_color_radius_feature(masked_image, color, radius))
        feature_two.append(get_length_of_lines_feature(masked_image, color))
        feature_three.append(measure_increase_in_connectivity_with_radius_feature(image, color, radius))

        
        datum = [color_features, status]
        data.append(datum)



    # save to text
    pickle.dump(data, open("data.txt", 'wb'))
    return data


def generate_image_based_on_values(image, color, radius):
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
        masked_image = generate_image_based_on_values(image, color, radius)
        lengths.append(get_length_of_lines_feature(masked_image, color))
    return stats.linregress(radii, lengths)[0]


def get_length_of_lines_feature(image, color):
    # use a probablistic hough line algorithm to calculate the average length of the lines detected
    edges = canny(image)
    lines = probabilistic_hough_line(edges, threshold=10, line_length=10, line_gap=3)
    return calculate_length_of_lines(lines)


def calculate_length_of_lines(lines):
    '''
    Given a list of a startpoints and endpoints for the lines, calculate the average length of the lines
    '''
    if len(lines) == 0:
        return 0
    else:
        line_length_sum = 0
        # iterate through each line
        for line in lines:
            p0, p1 = line
            # sum up the line length
            line_length_sum += sqrt((p0[0] - p1[0])**2 + (p0[1] - p1[1])**2)
        # average over the length of the line
        return float(line_length_sum) / len(lines)
