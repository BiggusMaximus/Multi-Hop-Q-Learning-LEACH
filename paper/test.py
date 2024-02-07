# %%
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1.inset_locator import zoomed_inset_axes
from mpl_toolkits.axes_grid1.inset_locator import mark_inset
import random
from matplotlib.ticker import (MultipleLocator, AutoMinorLocator, ScalarFormatter)
import matplotlib.patches as patches
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.path import Path
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler
from sklearn.datasets import make_blobs
import networkx as nx
import winsound
import time
import pandas as pd
from scipy import interpolate
np.seterr(divide='ignore', invalid='ignore')
import warnings
warnings.simplefilter("error")
from scipy.interpolate import griddata
import matplotlib

class OOMFormatter(matplotlib.ticker.ScalarFormatter):
    def __init__(self, order=0, fformat="%1.1f", offset=True, mathText=True):
        self.oom = order
        self.fformat = fformat
        matplotlib.ticker.ScalarFormatter.__init__(self,useOffset=offset,useMathText=mathText)
    def _set_order_of_magnitude(self):
        self.orderOfMagnitude = self.oom
    def _set_format(self, vmin=None, vmax=None):
        self.format = self.fformat
        if self._useMathText:
            self.format = r'$\mathdefault{%s}$' % self.format

df = pd.read_excel('./EPOCH_1000_0.1_QL_tuning.xlsx')

fig = plt.figure(figsize=(10, 8))
ax = fig.add_subplot(111, projection='3d')

# Scatter plot the points
ax.scatter(df['ALPHA'], df['GAMMA'], df['ENERGY'], c='blue', marker='o', label='Data Points')

# Create a meshgrid for the surface plot
resolution = 1000
X, Y = np.meshgrid(np.linspace(df['ALPHA'].min(), df['ALPHA'].max(), resolution), 
                   np.linspace(df['GAMMA'].min(), df['GAMMA'].max(), resolution))
# Z = np.array(df['ENERGY']).reshape(X.shape)
Z = griddata((df['ALPHA'], df['GAMMA']), df['ENERGY'], (X, Y), method='cubic')
# Plot the 3D surface
surf = ax.plot_surface(X, Y, Z, cmap='viridis', alpha=0.7, edgecolor='k', linewidth=0.5)

# Add labels and title
ax.set_xlabel(r'${α}   $', fontname='Times New Roman'           , fontsize=10)
ax.set_ylabel(r'${γ}   $', fontname='Times New Roman'           , fontsize=10)
ax.set_zlabel(r'$Energy\, (J)     $', fontname='Times New Roman', fontsize=10)
ax.tick_params(labelsize=8)

# ax.ticklabel_format(axis='z', style='sci', scilimits=(2,5))
formatter = ScalarFormatter(useMathText=True)
formatter.set_scientific(True)
formatter.set_powerlimits((-2, 2))  # Adjust as needed
ax.zaxis.set_major_formatter(formatter)
ax.zaxis.get_major_formatter().set_scientific(True)
# ax.zaxis.set_major_locator(MultipleLocator(1e5))
# ax.zaxis.set_major_formatter('{x:1.2f}')
# ax.zaxis.set_minor_locator(MultipleLocator(2.5e4))

ax.yaxis.set_major_locator(MultipleLocator(0.2))
ax.yaxis.set_major_formatter('{x:.2f}')
ax.yaxis.set_minor_locator(MultipleLocator(0.05))

ax.xaxis.set_major_locator(MultipleLocator(0.2))
ax.xaxis.set_major_formatter('{x:.2f}')
ax.xaxis.set_minor_locator(MultipleLocator(0.05))

# Show the plot

min_indices = np.unravel_index(np.argmin(Z), Z.shape)

# Get the corresponding X, Y, and Z values
min_x = X[min_indices]
min_y = Y[min_indices]
min_z = Z[min_indices]

print(f"The values of X, Y, and Z when Z is at its lowest are: {min_x}, {min_y}, {min_z}")

plt.show()