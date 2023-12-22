import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1.inset_locator import zoomed_inset_axes
from mpl_toolkits.axes_grid1.inset_locator import mark_inset
import random
from matplotlib.ticker import (MultipleLocator, AutoMinorLocator)
import matplotlib.patches as patches
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.path import Path

def triangular_grid2d(l, h, r):
    d = r*np.sqrt(3)
    N_l = np.ceil((((l-d)/d) + 1))
    N_h = np.ceil(((2*np.sqrt(3)*h-6*d+4*np.sqrt(3)*r)/(3*d)) + 1)
    print(N_h)
    N_total = N_h * N_l
    dy = np.power(d, 2) - np.power(d/2, 2)
    dy = np.sqrt(dy)
    dh = ((np.sqrt(3)/2) * d) - r
    dl = d/2

    sensor_pos = {'x' : [], 'y':[]}
    sensor_coverage = []

    count = 0
    for x in range(0, int(N_l)):
        for y in range(0, int(N_h)):
            if y % 2 != 0:
                # Odd
                x_p = (d * x) 
                y_p = dy * y
                sensor_pos['x'].append(x_p)
                sensor_pos['y'].append(y_p)
                sensor_coverage.append((x_p, y_p))
            else:
                x_p = (d * x) + (d/2)
                y_p = dy * y
                sensor_pos['x'].append(x_p)
                sensor_pos['y'].append(y_p)
                sensor_coverage.append((x_p, y_p))

            count += 1
            
    print(f"N_i : {N_l} | N_h : {N_h} | N_total : {N_total} | d/r : {d/r} | count : {count} | d : {d} | dy : {dy} | dl : {dl} | dh : {dh}")
    return sensor_pos, sensor_coverage

def triangular_grid2d_inverse(l, h, r):
    d = r*np.sqrt(3)
    N_l = np.ceil((((l-d)/d) + 1))
    N_h = np.ceil(((2*np.sqrt(3)*h-6*d+4*np.sqrt(3)*r)/(3*d)) + 1)
    print(N_h)
    N_total = N_h * N_l
    dy = np.power(d, 2) - np.power(d/2, 2)
    dy = np.sqrt(dy)
    dh = ((np.sqrt(3)/2) * d) - r
    dl = d/2

    sensor_pos = {'x' : [], 'y':[]}
    sensor_coverage = []

    count = 0
    for x in range(0, int(N_h)):
        for y in range(0, int(N_l)):
            if y % 2 != 0:
                # Odd
                x_p = (d * x) 
                y_p = dy * y
                sensor_pos['x'].append(x_p)
                sensor_pos['y'].append(y_p)
                sensor_coverage.append((x_p, y_p))
            else:
                x_p = (d * x) + (d/2)
                y_p = dy * y
                sensor_pos['x'].append(x_p)
                sensor_pos['y'].append(y_p)
                sensor_coverage.append((x_p, y_p))

            count += 1
            
    print(f"N_i : {N_l} | N_h : {N_h} | N_total : {N_total} | d/r : {d/r} | count : {count} | d : {d} | dy : {dy} | dl : {dl} | dh : {dh}")
    return sensor_pos, sensor_coverage

def triangular_grid3d(l, h, depth, r):
    d = r*np.sqrt(3)
    N_l = np.ceil((((l-d)/d) + 1))
    N_h = np.ceil(((2*np.sqrt(3)*h-6*d+4*np.sqrt(3)*r)/(3*d)) + 1)
    N_d = np.ceil(((2*np.sqrt(3)*depth-6*d+4*np.sqrt(3)*r)/(3*d)) + 1)

    N_total = N_h * N_l
    dy = np.power(d, 2) - np.power(d/2, 2)
    dy = np.sqrt(dy)
    dh = ((np.sqrt(3)/2) * d) - r
    dl = d/2

    sensor_pos = {'x' : [], 'y':[], 'z':[]}
    sensor_coverage = []

    count = 0
    for z in range(0, int(N_d)):
        for x in range(0, int(N_h)):
            for y in range(0, int(N_l)):
                if z % 2 == 0:
                    if y % 2 != 0:
                        # Odd
                        x_p = (d * x) 
                        y_p = dy * y
                        z_p = dy * z
                        sensor_pos['x'].append(x_p)
                        sensor_pos['y'].append(y_p)
                        sensor_pos['z'].append(z_p)
                        sensor_coverage.append((x_p, y_p, z_p))
                    else:
                        x_p = (d * x) + (d/2)
                        y_p = dy * y
                        z_p = dy * z
                        sensor_pos['x'].append(x_p)
                        sensor_pos['y'].append(y_p)
                        sensor_pos['z'].append(z_p)
                        sensor_coverage.append((x_p, y_p, z_p))
                else:
                    if y % 2 != 0:
                        # Odd
                        x_p = (d * x) + (d/2)
                        y_p = dy *  + (d/2)
                        z_p = dy * z
                        sensor_pos['x'].append(x_p)
                        sensor_pos['y'].append(y_p)
                        sensor_pos['z'].append(z_p)
                        sensor_coverage.append((x_p, y_p, z_p))
                    else:
                        x_p = (d * x) + (d/2) + (d/2)
                        y_p = dy * y + (d/2)
                        z_p = dy * z
                        sensor_pos['x'].append(x_p)
                        sensor_pos['y'].append(y_p)
                        sensor_pos['z'].append(z_p)
                        sensor_coverage.append((x_p, y_p, z_p))
                
            
    print(f"N_i : {N_l} | N_h : {N_h} | N_total : {N_total} | d/r : {d/r} | count : {count} | d : {d} | dy : {dy} | dl : {dl} | dh : {dh}")
    return sensor_pos, sensor_coverage

h = 100
l = 100
depth =100
r = 10


def get_cube():   
    phi = np.arange(1,10,2)*np.pi/4
    Phi, Theta = np.meshgrid(phi, phi)

    x = np.cos(Phi)*np.sin(Theta)
    y = np.sin(Phi)*np.sin(Theta)
    z = np.cos(Theta)/np.sqrt(2)
    return x,y,z

a = 100
b = 100
c = -10
x,y,z = get_cube()

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')


sensor_pos1, sensors_coverage1 = triangular_grid2d(l, h, r)
# Scatter plot with specified radii
ax.scatter(h/2, l/2, 15, label='Dependent Node', s=65, marker="s", color='r', edgecolors='k')
ax.scatter(sensor_pos1['x'], sensor_pos1['y'], -10, label='Base station', marker="o",color='m', edgecolors='k')

ax.plot_surface((x*a) + 50, (y*b) + 50, (z*c) - 5, alpha=0.2)

ax.set_xlabel('X Label')
ax.set_ylabel('Y Label')
ax.set_zlabel('Z Label')
ax.set_zlim([-10, 50])

ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.05),
              fancybox=True, shadow=True, ncol=5, markerscale=1, fontsize=10)

font = {
        'color':  'black',
        'weight': 'bold'
        }

ax.set_title('3D Deployment UWSN', fontdict=font)
ax.set_xlabel('Depth (m)')
ax.set_ylabel('Width (m)')
ax.set_zlabel('Length (m)')

plt.show()

sensor_pos, sensors_coverage = triangular_grid3d(l, h, depth, r)
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

# Scatter plot with specified radii
ax.scatter(sensor_pos['x'], sensor_pos['y'], sensor_pos['z'], s=r)

ax.set_xlabel('Depth (m)')
ax.set_ylabel('Width (m)')
ax.set_zlabel('Length (m)')

plt.show()
