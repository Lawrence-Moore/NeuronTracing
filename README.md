# NeuronTracing Tools

The repo contains tools to both manually and automatically select color regions to isolate neurons and aid in tracing. The code is written in python and uses numpy, scipy, and PyQT among other popular packages.

## The Basics

In Code/Algorithms/helper_functions.py, several methods exist to perform basic tasks like reading and saving images, generating MIPs, and visualizing images. 

- **Reading Images**. Most image stacks are in CZI format. To read such files, use the read_czi_file function. It takes in a string of the file name (including extension) and returns a list of numpy arrays representing each image layer.

    ```images = read_czi_file("example.czi")```

- **Saving Images**. To save a list of images representing an image strack, use save_images. The first input is the list of images, the second is the name of the images. The function will store each as a tiff with the z-layer attached to the name.

    ```save_images(images, "beautiful_image)```

- **Generating a Maximum Intensity Projection**. To generate a mip, use the function generate_mip. It takes in a list of 3D numpy arrays representing images and returns a single 3D numpy array representing the MIP

    ```mip = generate_mip(images)```

- **Display an image**. To display an image represented by a numpy array that's in 16 bytes, use the display_image function.

- **Visualizing a Mask**. To visualize mask on a 2D image, use the visualize_mask function. Given a 2D numpy array (image) and a mask with the same size, it will display the masked image.

    ```visualize_mask(image, mask)```

## Image Normalization

The code to normalize the colors in an image resides in code/Algorithms/image_normalization.py. There are two options when normalizing: consider the entire image or only a thresholded segment.

- **Normalize without threshold**. To normalize an image without a threshold, use the normalize_generic function. This function takes in one input, namely the list of images representing the image stack.
- **Normalize with threshold**. Use the normalize_with_standard_deviation function. The function takes two inputs: the list of images representing the image stack and a number representing the std_multiple. The method then normalizes the image using only the pixels greater than the mean and less than the mean plus std_multiple * standard devation.

To normalize, the function finds the mean and standard deviation of the whole sample and then goes through each layer and adjusts its mean and standard to match those of the entire stack. When there's a threshold, this is only done on the pixels meeting the specified range.

To threshold, a mask is generated for each layer independently. If any one pixel meets the threshold for some color layer, it's included in the overall mask.

## Alignment

The code to align the color layers resides in code/Algorithms/alignment.py. The method responsible for aligning has the follow signature:

    align_images(images, wiggle_room=20, manual=False, template_top_left_x=0,
                                         template_top_left_y=0,
                                         template_width=0,
                                         template_color_layer=0,
                                         template_image_index=0)

Paramters:

- Images is a list of numpy arrays.
- wiggle_room is how much a color layer can be moved in the x, y, or z direction.
- manual is whether the region used to align the color layers is specified by the user or chosen automatically.
- template_top_left_x is the x coordinate of top left point of the manually selected region.
- template_top_left_y is the y coordinate of top left point of the manually selected region.
- template_width is the width of the template.
- template_color_layer is the index of the color layer used as the template (either 0, 1, 2 representing red, green, or blue).
- template_image_index is the index of the image to which the template belongs.

To align the colors, a region (or patch as its often referred to in the code) is selected from one color layer at a specific z layer. This region serves as a template to be used in template matching, and using the template matching implementation from skimage, this region is found in the other two colors layers. The offset from its original position to the position of the template in the other layers is recorded and used to shift the layers into alignment.

## K-Means
For K-Means clustering, the core implementation in [sklearn](http://scikit-learn.org/) is used. The method that handles using k-means is found in code/Algorithms/clustering.py. The method signature looks as follows
    
    k_means(image=None, images=None, weights=None, n_colors=64, num_training=1000, std_multiple=0,
            threshold=False, show_plot=False, show_color_space_assignment=False)

Parameters:
- Given an image or stack of images (list of np arrays), perform k means and return a list of the color centers
- num_training is the number of pixels used to train
- weights is a list of rgb values to initialize the algorithm with
- n_colors is the number of clusters
- threshold is a boolean whether to consider the black pixels or not
- std_multiple is the standard deviation multiple used in thresholding
- show_plot is a boolean whether to show the quantized image or not
- show_color_space_assignment is a boolean whether to show the way the color space is clustered or not

The code is well documented. The only thing that might appear unclear is the way the pixels are selected, which is addressed in the sampling data section below.

## Self Organizing Map
The core implementation of the SOM is from this [repo](https://github.com/JustGlowing/minisom). I added a few methods, such as the ability to pass in weights, train in batch, and the ability to return the specific index of the winning neuron. The code (with changes) is in code/Algorithms.minisom.py.

Things to note: 
- weights are stored in a MxNx3 format, where MxN is the layout of the nodes.
- having a weight of all zeros will cause things to break, as the weight is normalized after each step. Considering we don't ever want an all black dominant color, I left this in as a caution of something potentially going wrong.
- the code is well documented, so after briefly reading through it the implementation will make sense

The enclosing method to cluster an image is in the same file as the k-means one: code/Algorithms/clustering.py. It takes the same parameters with one additional: weights_max_value, which is used to normalize a weight vector to 0 - 1. 

Given the number of colors, I find the largest two factors so that I can arrange the grid of nodes in as close a shape as possible to a perfect square. One major thing to note is that **as it stands now, you can't use an prime number of color clusters unless you make them line up along a single axis**. The algorithm seems to give better results when the nodes are arranged in as close to a perfect square as possible, so if the user specifies a prime number of colors, **I bump up the number by one so that I can have an even grid**. If it's important to have the exact number of colors and its prime, it'd would be easy to write the code to override this.

## Sampling Data For Clustering

As it stands now, every clustering algorithm has several options for the pixels on which it trains. You can either choose colors from the MIP or the entire stack. In addition, you can train on just randomly selected colors or pixels that meet a threshold. The method sample_data in clustering.py handles all these cases.


## Visualizing in the Color Space

In clustering.py, the function visualize_color_space allows the user to see how the color space is separated by k-means and SOM. The method takes in either the kmeans or som object that contain the center of the clusters along with the intensity value (on a scale of 0 to 1). It then generates a pretty plot of the color space separation. 


## SVM

The overall pipeline using SVM is currently somewhat unfinished. The methods to extract features work. All that needs to be done is a bit of preprocessing on the data so that every feature has a mean of 0 and standard deviation of 1. Further, quality training data needs to be generated and then the SVM can be trained. 

It's worth noting that the SVM is a binary classifier, simply stating whether a color is present or not in the image. This means that separating the image into discrete colors won't be immediately possible, as the algorithm isn't selecting color regions per sé but just color pixels that should be present. Something to fool around with in the future though.



