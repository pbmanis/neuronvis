from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.ticker import LinearLocator, FormatStrFormatter
import numpy as np
from scipy.linalg import norm

class TruncatedCone(object):
    def __init__(self, facets=12):
        self.facets = facets
    
    def make_truncated_cone(self, p0=None, p1=None, R=[1.0, 1.0]):

        # -*- coding: utf-8 -*-
        """
        Created on Sun Oct  2 18:33:10 2016

        Modified from 
        https://stackoverflow.com/questions/38076682/how-to-add-colors-to-each-individual-face-of-a-cylinder-using-matplotlib
        to add "end caps" and to undo fancy coloring.

        @author: astrokeat
        """

        # #axis and radius
        # p0 = np.array([1, 3, 2]) #point at one end
        # p1 = np.array([8, 5, 9]) #point at other end
        # R = 5

        #vector in direction of axis
        v = p1 - p0

        #find magnitude of vector
        mag = norm(v)

        #unit vector in direction of axis
        v = v / mag

        #make some vector not in the same direction as v
        not_v = np.array([1, 0, 0])
        if (v == not_v).all():
            not_v = np.array([0, 1, 0])

        #make vector perpendicular to v
        n1 = np.cross(v, not_v)
        #normalize n1
        n1 /= norm(n1)

        #make unit vector perpendicular to v and n1
        n2 = np.cross(v, n1)

        #surface ranges over t from 0 to length of axis and 0 to 2*pi
        t = np.linspace(0, mag, num=2, endpoint=True)
        theta = np.linspace(0, 2 * np.pi, self.facets)
        rsamplet = np.linspace(0, R[0], 2, endpoint=True)
        rsampleb = np.linspace(0, R[1], 2, endpoint=True)

        #use meshgrid to make 2d arrays
        t, theta2 = np.meshgrid(t, theta)

        rsamplet,theta = np.meshgrid(rsamplet, theta)
        rsampleb,theta = np.meshgrid(rsampleb, theta)

        #generate coordinates for surface
        # "Tube"
        X, Y, Z = [p0[i] + v[i] * t + R * np.sin(theta2) * n1[i] + R * np.cos(theta2) * n2[i] for i in [0, 1, 2]]
        # end caps
        # "Bottom"
        X2, Y2, Z2 = [p0[i] + rsamplet[i] * np.sin(theta) * n1[i] + rsamplet[i] * np.cos(theta) * n2[i] for i in [0, 1, 2]]
        # "Top"
        X3, Y3, Z3 = [p0[i] + v[i]*mag + rsampleb[i] * np.sin(theta) * n1[i] + rsampleb[i] * np.cos(theta) * n2[i] for i in [0, 1, 2]]
        return((X, Y, Z), (X2, Y2, Z2), (X3, Y3, Z3))

def plot_tc(p0=None, p1=None, R=[0., 0.], c='blue'):
    C, T, B = hcyl.make_truncated_cone(p0, p1, R)
    ax=plt.subplot(111, projection='3d')
    ax.plot_surface(C[0], C[1], C[2], color=c, linewidth=1, antialiased=False)
    ax.plot_surface(T[0], T[1], T[2], color=c)
    ax.plot_surface(B[0], B[1], B[2], color=c)    

if __name__ == '__main__':
    hcyl = TruncatedCone()

    plot_tc(p0=np.array([1, 3, 2]), p1=np.array([8, 5, 9]), R=[5.0, 2.0])


    plt.show()