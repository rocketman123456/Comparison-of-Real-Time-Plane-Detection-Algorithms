import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
normals = []
points = []
vertices = []
with open('planeIGuess.txt', 'r') as file:
    for line in file.readlines():
        n, c, v = line.split(';')
        n = n.replace('Normal: ', '')[1:-1].split(', ')
        c = c.replace('Center: ', '')[2:-1].split(', ')
        normals.append(np.array([float(en) for en in n]))
        points.append(np.array([float(cn) for cn in c]))
        v = v.replace('Vertices: [[', '').replace(' ', '')[:-3].split('],[')
        print(v)


fig = plt.figure()
# a plane is a*x+b*y+c*z+d=0
# [a,b,c] is the normal. Thus, we have to calculate
# d and we're set
for point, normal in zip(points, normals):
    d = -point.dot(normal)
    # create x,y
    xx, yy = np.meshgrid(range(10), range(10))

    # calculate corresponding z
    z = (-normal[0] * xx - normal[1] * yy - d) * 1. / normal[2]

    # plot the surface
    plt3d = fig.gca(projection='3d')
    plt3d.plot_surface(xx, yy, z)
# plt.show()
