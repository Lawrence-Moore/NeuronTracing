import numpy as np
from skimage.measure import structural_similarity as ssim
from skimage.feature import match_template
from matplotlib import pyplot as plt


def align_images(images, wiggle_room=20, manual=False, template_top_left_x=0,
                                         template_top_left_y=0,
                                         template_width=0,
                                         template_color_layer=0,
                                         template_image_index=0):
    """
    Given a list of np arrays representing images, this method aligns the color layers

    PARAMS:
    wiggle_room is how much a color layer can be moved in the x, y, or z direction
    manual is whether a region is specified by the user or chosen automatically
    template_top_left_x is the x coordinate of top left point of the manually selected region
    template_top_left_y is the y coordinate of top left point of the manually selected region
    template_width is the width of the template
    template_color_layer is the index of the color layer used as the template (either 0, 1, 2 representing red, green, or blue)
    template_image_index is the index of the image to which the template belongs
    """
    # align blue and green and then blue and red
    if manual is True:
        patch_coordinate = (template_top_left_x, template_top_left_y)

        # retrieve the template
        patch = images[template_image_index][:, :, template_color_layer][template_top_left_x: template_top_left_x + template_width,
                                                                         template_top_left_y: template_top_left_y + template_width]
        color_indexes = [0, 1, 2]
        color_indexes.remove(template_color_layer)

        # shift over the necessary color indexes
        aligned_images, offsets1 = adjust_color_layer(images, color_indexes[0], patch_coordinate, patch, template_image_index, template_width, wiggle_room=wiggle_room)
        aligned_images, offsets2 = adjust_color_layer(aligned_images, color_indexes[1], patch_coordinate, patch, template_image_index, template_width, wiggle_room=wiggle_room)

        # make sure the template index is correct after shifting in the z direction
        if template_image_index >= len(aligned_images):
            template_image_index = len(aligned_images) - 1

        visualize_alignment(images, aligned_images, patch_coordinate, template_image_index, template_width, color_indexes, offsets1, offsets2)
        return aligned_images
    else:
        # choose an image from them iddle
        image_index = len(images) / 2
        image = images[image_index]

        # find the brightest patch in any of the color layers
        width = 50
        patch_coordinate, best_color, patch = get_bright_patch(image, width)

        color_indexes = [0, 1, 2]
        color_indexes.remove(best_color)

        # shift the color layers
        aligned_images, offsets1 = adjust_color_layer(images, color_indexes[0], patch_coordinate, patch, image_index, width)
        aligned_images, offsets2 = adjust_color_layer(aligned_images, color_indexes[1], patch_coordinate, patch, image_index, width)
        if template_image_index >= len(aligned_images):
            template_image_index = len(aligned_images) - 1  # make sure the template index is correct after shifting in the z direction
        visualize_alignment(images, aligned_images, patch_coordinate, image_index, width, color_indexes, offsets1, offsets2)

        return aligned_images


def adjust_color_layer(images, color_index, patch_coordinate, patch, image_index, width, wiggle_room=20):
    '''
    Adjust the color layers so they align as closely as possible

    color_index refers to the color layer we're trying to adjust (if the patch is red for example, this would be green or blue)
    patch_coordinate is a tuple of the top left coordinate of the patch
    patch is a numpy array of representing the image of the patch
    image_index is the index of the layer to which the patch belongs
    width is the width of the template
    '''

    # find that patch in blue and compare coordinates
    offsets, best_patch = match_patch(images, patch_coordinate, patch, width, image_index, color_index, wiggle_room=wiggle_room)

    # visualize the match
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

            # find the minimum std
            if np.min(stds) > highest_minimum_std:
                highest_minimum_std = np.min(stds)
                coordinates = [i, j]
                best_color = np.argmin(stds)

    # get the patch
    patch = image[:, :, best_color][coordinates[0]: coordinates[0] + width, coordinates[1]: coordinates[1] + width]

    # subtract width to get the top left of the square
    return (coordinates, best_color, patch)


def match_patch(images, color_index, patch_coordinate, patch, image_index, width, wiggle_room=20):
    '''
    Given the same input as adjust_color_layer above,
    it returns the offset of the best match of the patch in this color layer along with the matched patch
    '''
    result = 0
    final_z_offset = 0

    # Go through and find the greatest similarity in the layers

    # make sure the shifts happen within the right range
    offset = wiggle_room
    min_shift = -1 * min(offset, image_index)
    max_shift = min(offset + image_index, len(images)) - image_index

    for z_offset in range(min_shift, max_shift):
        image = images[image_index + z_offset]

        # get the region to check. It should be the size of the patch plus some wiggle room around the edges
        comparison_area = image[patch_coordinate[0] - wiggle_room: patch_coordinate[0] + width + wiggle_room,
                                patch_coordinate[1] - wiggle_room: patch_coordinate[1] + width + wiggle_room, color_index]
        comparison_area = np.reshape(comparison_area, (wiggle_room * 2 + width, wiggle_room * 2 + width))

        # perform the template matching
        new_result = match_template(comparison_area, patch)
        if np.max(new_result) > np.max(result):
            result = new_result
            final_z_offset = z_offset

    xy = np.unravel_index(np.argmax(result), result.shape)

    # adjust the coordinates
    xy_offset = np.array(xy) - wiggle_room
    best_patch = images[image_index + final_z_offset][patch_coordinate[0] + xy_offset[0]: patch_coordinate[0] + xy_offset[0] + width,
                                                      patch_coordinate[1] + xy_offset[1]: patch_coordinate[1] + xy_offset[1] + width, color_index]

    return (xy_offset[0], xy_offset[1], final_z_offset), best_patch


def alternative_patch_matching(images, patch_coordinate, patch, width, image_index, color_index, wiggle_room=5):
    '''
    Uses image comparison instead of fuzzy matching
    '''
    best_result = 0
    final_z_offset = 0
    for z_offset in range(-20, 20):
        image = images[image_index + z_offset]
        # iterate through squares and check if the images are similar
        for i in range(patch_coordinate[0] - wiggle_room, patch_coordinate[0] + wiggle_room):
            for j in range(patch_coordinate[1] - wiggle_room, patch_coordinate[1] + wiggle_room):
                comparison_area = image[i: i + width, j: j + width, color_index]
                comparison_score = ssim(patch, comparison_area)
                if comparison_score > best_result:
                    best_result = comparison_score
                    final_z_offset = z_offset
                    best_xy = [i, j]

    xy_offset = np.array(best_xy) - patch_coordinate
    best_patch = images[image_index + final_z_offset][patch_coordinate[0] + xy_offset[0]: patch_coordinate[0] + xy_offset[0] + width,
                                                      patch_coordinate[1] + xy_offset[1]: patch_coordinate[1] + xy_offset[1] + width, color_index]

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

        shifted_images = shifted_images[:len(shifted_images) - abs(offset[2])]

    elif offset[2] < 0:
        # shift the z layer of the chanel down
        for i in reversed(range(abs(offset[2]), len(shifted_images))):
            shifted_images[i][:, :, color_chanel] = shifted_images[i - abs(offset[2])][:, :, color_chanel]

        shifted_images = shifted_images[abs(offset[2]):]

    return shifted_images


def visualize_alignment(images, aligned_images, patch_coordinate, image_index, width, color_indexes, offsets1, offsets2):
    '''
    Compare the patch before and after in the images
    '''
    old_patch = images[image_index][patch_coordinate[0]: patch_coordinate[0] + width, patch_coordinate[1]: patch_coordinate[1] + width, :]
    new_patch = aligned_images[image_index][patch_coordinate[0]: patch_coordinate[0] + width, patch_coordinate[1]: patch_coordinate[1] + width, :]

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
