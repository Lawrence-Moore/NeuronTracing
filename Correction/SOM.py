import numpy as np
from scipy.spatial.distance import euclidean
from random import randint
from math import exp, log


class SOM(object):

    def __init__(self, image, num_nodes, num_dimensions=3, initial_weight_vectors=[]):
        # set up the SOM for greatness
        self.image = image

        # go through and create nodes
        self.nodes = []
        if initial_weight_vectors == []:
            for _ in range(num_nodes):
                random_vector = np.random.random(num_dimensions) * np.max(image)
                initial_weight_vectors.append(random_vector)

        for position, vector in zip(range(num_nodes), initial_weight_vectors):
            self.nodes.append(Node([position, 0], weight_vector=vector))

    def gaussian_neighborhood_function(self, node1, node2, sigma):
        '''
        Calculates the neighborhood multiplyer value in the update stepusing the guassian function
        '''
        return exp(-1 * euclidean(node1.coordinates, node2.coordinates) ** 2 / (2 * sigma ** 2))

    def find_most_similar_node(self, input_value):
        '''
        Find the closest node.
        Takes in a numpy array representing a sample from the training data
        returns the node that's closest using the euclidean distance metric
        '''
        min_node_index = np.argmin([euclidean(input_value, node.weight_vector) for node in self.nodes])
        return self.nodes[min_node_index]

    def learn(self, neighborhood_function, initial_sigma=5.0, max_num_iterations=100, learning_rate=0.1):
        sigma = initial_sigma
        for time in range(max_num_iterations):
            # randomly sample from the image
            random_vector = self.image[randint(0, self.image.shape[0] - 1), randint(0, self.image.shape[1] - 1)]

            # find BMU (best matching unit)
            bmu = self.find_most_similar_node(random_vector)

            # update everything accordingly
            for node in self.nodes:
                node.weight_vector = node.weight_vector + learning_rate * neighborhood_function(bmu, node, sigma) * (node.weight_vector - random_vector)

            learning_rate = learning_rate * exp(-1 * time/max_num_iterations)
            sigma = sigma * exp(-1 * log(initial_sigma) / max_num_iterations)


class Growing_Som:
    pass


class Node(object):

    def __init__(self, coordinates, weight_vector):
        self.coordinates = coordinates
        self.weight_vector = weight_vector

    def __str__(self):
        return "Node at (%d, %d) with weight vector [%d, %d, %d]" \
                    % (self.coordinates[0], self.coordinates[1], self.weight_vector[0], self.weight_vector[1], self.weight_vector[2])

    __repr__ = __str__


def cluster_image(image, som):
    new_image = np.zeros(image.shape)
    width, height, _ = image.shape
    for i in range(width):
        for j in range(height):
            # get the pixel
            pixel = image[i, j]

            # get the closest color
            closest_color = som.find_most_similar_node(pixel).weight_vector
            new_image[i, j] = closest_color
    return new_image
