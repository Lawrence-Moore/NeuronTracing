from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from scipy import ndimage
from matplotlib import cm
from matplotlib.ticker import LinearLocator, FormatStrFormatter
import matplotlib.pyplot as plt
import numpy as np
import cv2
from  scipy.misc import imresize


def displayValidityMap(map, compression, size):
    '''
    :param map: numpy 3D boolean array: colorspace with selected region as True
    :param compression: int: downsampling ratio of map w/ shape=(z, x, y)
    :param size: int: size of scatter points
    :return: plots the surface of the colorspace in 3D by downsampling the
    space, dilating the region within by one pixel in all directions,
    subtracting the original region, and then scatter plotting the remaining
    points with matplotlib
    '''
    if type(map) is bool:
        return
    map = np.array(map)
    vlength, xlength, ylength = map.shape
    # create an empty colorspace that is downsampled/compressed
    compressed = np.zeros((vlength/compression, xlength/compression, ylength/compression), dtype=bool)
    # fill it with values from map
    for x in xrange(0, xlength, compression):
        for v in xrange(compression, vlength-compression-1, compression):
            for y in xrange(1, ylength, compression):
                if map[v][x][ylength-y]:
                    compressed[v/compression][x/compression][y/compression] = True
    map = compressed
    fig = plt.figure()
    ax = Axes3D(fig)
    # dilate by 1 pixel in all directions
    dilationStruct = np.array([[[True] * 3] * 3] * 3)
    dilatedMap = ndimage.binary_dilation(map, structure=dilationStruct)
    surface = dilatedMap * (~ map)  # subtract the original map from dilated
    V, X, Y = np.nonzero(surface)  # obtain non-zero indices in map
    ax.set_zlim(0, vlength / compression)
    ax.set_xlim(0, xlength / compression)
    ax.set_ylim(0, ylength / compression)
    # plot with particles with color proportional to V, full opacity, and size
    ax.scatter(X, Y, V, c=V, alpha=1, s=size)
    plt.show()

def displayValidityMapFull3D(map, compression):
    # This function is deprecated and no longer in use
    '''
    :param map: numpy 3D boolean array: colorspace with selected region as True
    :param compression: int: downsampling ratio of map
    :return: plots the surface of the colorspace seen from the top
    '''
    if type(map) is bool:
        return
    map = np.array(map)
    vlength, xlength, ylength = map.shape
    # map = imresize(map, (vlength / compression, xlength/2, ylength/2))
    # map = map.astype(np.uint8)
    # xlength /= compression
    # ylength /= compression
    # vlength /= compression
    X, Y, V = np.array([]), np.array([]), np.array([])
    maxx, minx = 0, 255
    for x in xrange(0, xlength):
        for v in xrange(0, vlength):
            for y in xrange(0, ylength):
                if map[v*compression][x*compression][y*compression]:
                    X = np.append(X, x)
                    Y = np.append(Y, (ylength-y))
                    V = np.append(V, v)

                    if y > maxy:
                        maxy = y
                    if y < miny:
                        miny = y
                    if x > maxx:
                        maxx = x
                    if x < minx:
                        minx = x
            if maxy != 0 and miny != 255:
                X = np.append(X, [x, x])
                Y = np.append(Y, [miny, maxy])
                V = np.append(V, [v, v])
    for x in [minx, maxx]:
        for v in xrange(0, vlength):
            for y in xrange(0, ylength):
                if map[v*compression][x*compression][y*compression]:
                    X = np.append(X, x)
                    Y = np.append(Y, (ylength-y))
                    V = np.append(V, v)

    #ax.plot_surface(X, Y, V)
    #R = np.sqrt(X**2 + Y**2)
    #Z = np.sin(R)
    #ax.plot_wireframe(X, Y, V)
    #ax.plot_surface(X, Y, V, rstride=1, cstride=1, color='b')
    fig = plt.figure()
    ax = Axes3D(fig)
    surf = ax.plot_surface(X, Y, V, rstride=5, cstride=5, cmap=cm.coolwarm,
                           linewidth=0, antialiased=False)
    ax.set_zlim(-1, vlength+1)
    ax.zaxis.set_major_locator(LinearLocator(10))  # number of values in legend
    ax.zaxis.set_major_formatter(FormatStrFormatter('%d'))  # z-accuracy
    fig.colorbar(surf, shrink=0.5, aspect=5)
    plt.show()

