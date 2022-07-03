import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
normals = []
points = []
vertices = []
with open('planeIGuess.txt', 'r') as file:
    for line in file.readlines():
        l = line.split(' ')
        # n = n.replace('Normal: ', '')[1:-1].split(', ')
        n = [float(x.replace(',', '.')) for x in l[:-1]]
        # c = c.replace('Center: ', '')[2:-1].split(', ')
        d = float(l[-1].replace(',', ' '))
        # normals.append(np.array([float(en) for en in n]))
        normals.append(np.array(n))
        points.append(d)
        # v = v.replace('Vertices: [[', '').replace(' ', '')[:-3].split('],[')


fig = plt.figure()
# a plane is a*x+b*y+c*z+d=0
# [a,b,c] is the normal. Thus, we have to calculate
# d and we're set
for abc, d in zip(normals, points):
    a, b, c = abc
    # d = -point.dot(normal)
    # create x,y
    xx, yy = np.meshgrid(range(10), range(10))

    # calculate corresponding z
    z = (-a * xx - b * yy - d) * 1. / c

    # plot the surface
    plt3d = fig.gca(projection='3d')
    plt3d.plot_surface(xx, yy, z)
plt.show()
