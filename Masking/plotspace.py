from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits.mplot3d.art3d import Poly3DCollection

from matplotlib import cm
from matplotlib.ticker import LinearLocator, FormatStrFormatter
import matplotlib.pyplot as plt
import numpy as np

def displayValidityMap(map, compression, size): # v, x, y
    vlength, xlength, ylength = map.shape
    xlength /= compression
    ylength /= compression
    vlength /= compression
    fig = plt.figure()
    ax = Axes3D(fig)
    X, Y, V = np.array([]), np.array([]), np.array([])
    for x in xrange(0, ylength):
        for v in xrange(0, vlength):
            #maxy, miny = 0, 255
            for y in xrange(0, ylength):
                if map[v*compression][x*compression][y*compression]:
                    X = np.append(X, x)
                    Y = np.append(Y, (ylength-y))
                    V = np.append(V, v)
                    '''if y > maxy:
                        maxy = y
                    if y < miny:
                        miny = y
            if maxy != 0 and miny != 255:
                X = np.append(X, [x, x])
                Y = np.append(Y, [miny, maxy])
                V = np.append(V, [v, v])'''

    #ax.plot_surface(X, Y, V)
    #R = np.sqrt(X**2 + Y**2)
    #Z = np.sin(R)
    #ax.plot_wireframe(X, Y, V)
    #ax.plot_surface(X, Y, V, rstride=1, cstride=1, color='b')
    ax.scatter(X, Y, V, c=V, alpha=1, s=size)
    ax.set_zlim(0, vlength)
    ax.set_xlim(0, xlength)
    ax.set_ylim(0, ylength)
    '''surf = ax.plot_surface(X, Y, Z, rstride=5, cstride=5, cmap=cm.coolwarm,
                           linewidth=0, antialiased=False)
    ax.set_zlim(-1, vlength+1)

    ax.zaxis.set_major_locator(LinearLocator(10)) # number of values in legend
    ax.zaxis.set_major_formatter(FormatStrFormatter('%d')) # z-accuracy

    fig.colorbar(surf, shrink=0.5, aspect=5)'''

    plt.show()



'''
with tifffile.TIFFfile('img.tif') as tif:
    originalImage = tif.asarray()
if originalImage.dtype == np.uint16:
    originalImage /= 256
    originalImage = originalImage.astype(np.uint8)
if originalImage.shape[2] == 4:
    originalImage = originalImage[:, :, 0:3]

#originalImage = cv2.resize(originalImage, (400, 400))
hsv = cv2.cvtColor(originalImage, cv2.COLOR_RGB2HSV)
bgr = cv2.cvtColor(originalImage, cv2.COLOR_RGB2BGR)
#hsv = hsv.astype(np.float32)

indices = [[[[] for x in xrange(0, 256)] for y in xrange(0, 256)] for z in xrange(0, 256)]
h, s, v = [hsv[:, :, x] for x in xrange(3)]
height, width = hsv.shape[0], hsv.shape[1]
hsv = hsv.reshape((width * height), 3)

a = time.time()
for py in xrange(0, width*height):
        [g, t, q] = hsv[py]
        indices[g][t][q].append(py)
b = time.time()

print (b-a)*1000
a = time.time()
for x in xrange(0, 256):
    for y in xrange(0, 256):
        for v in xrange(0, 256):
            ind = indices[x][y][v]
            hsv[ind] = [0, 0, 0]
b = time.time()
print (b-a)*1000'''

'''q = time.time()
for a in xrange(0, hsv.shape[0]):
    for b in xrange(0, hsv.shape[1]):
        [h, s, v] = hsv[a][b]
r = time.time()
print (r-q)*1000


def serial(z):
    if z > 10:
        z = 5
    else:
        z = 4

parallel = np.vectorize(serial, otypes=[np.float32])
w = time.time()
hsv = parallel(hsv)
v = time.time()
print (v-w)*1000
print hsv'''


'''
# Threshold the HSV image to get only blue colors
mask = cv2.inRange(hsv, lower_blue, upper_blue)
d = time.time()
for i in xrange(0, 200):
    lower_blue[2] += 1
    upper_blue[2] += 1
    a = time.time()
    tempmask = cv2.inRange(hsv, lower_blue, upper_blue)
    b = time.time()
    mask = cv2.bitwise_or(mask, tempmask)
    c = time.time()
res = cv2.bitwise_and(frame,frame, mask= mask)
e = time.time()'''