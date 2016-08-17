import numpy as np
from matplotlib import pyplot as plt
from czifile import CziFile


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


def visualize_mask(image, mask):
    '''
    Given a 2D numpy array (image) and a mask with the same size, display the image
    '''
    img = image.copy()
    img[np.logical_not(mask)] = 0
    display_image(img)


def display_image(image):
    plt.imshow((image.astype(np.float16) / (2 ** 16-1) * (2 ** 8-1)).astype(np.uint8))
    plt.show()


def separate_colors(image, colors, image_name):
    '''
    Given an image (3D numpy array) and list of 3 element arrays representing colors, split it up into several different colors
    '''
    for index, color in enumerate(colors):
        img = image.copy()
        img[img != color] = 0
        save_image(img, image_name + " color " + str(index) + ".jpg")
