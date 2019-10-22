import matplotlib.pyplot as plt
from matplotlib import cm
from mpl_toolkits.mplot3d import Axes3D
import numpy as np


def draw(data, x, mean, upper, lower, title="figure"):
    plt.figure()
    plt.scatter(data[0], data[1], marker='.', color='#0000ba')
    plt.plot(x, mean)
    plt.plot(x, upper)
    plt.plot(x, lower)
    plt.title(title)
    plt.show()


def draw_surface(x, y, z):
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.plot_surface(x, y, z, cmap=cm.hot, )
    ax.set_xlabel('X Label')
    ax.set_ylabel('Y Label')
    ax.set_zlabel('Z Label')
    plt.show()


def sort_by_x(x, y):
    x = np.array(x)
    y = np.array(y)
    sort_seq = np.argsort(x)
    x = x[sort_seq]
    y = y[sort_seq]
    return x, y
