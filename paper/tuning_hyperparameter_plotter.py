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

df = pd.read_excel('./TRIAL_II_EPOCH_1000_GAMMA_0.5_EPSILON_[0.1-1]_QL_tuning.xlsx')

fig = plt.figure(figsize=(10, 8))
ax = fig.add_subplot(111, projection='3d')

# Scatter plot the points
ax.scatter(df['ALPHA'], df['EPSILON'], df['Energy Residual'], c='blue', marker='o', label='Data Points')

# Create a meshgrid for the surface plot
resolution = 1000

X, Y = np.meshgrid(np.linspace(df['ALPHA'].min(), df['ALPHA'].max(), resolution), 
                   np.linspace(df['EPSILON'].min(), df['EPSILON'].max(), resolution))
# Z = np.array(df['ENERGY']).reshape(X.shape)
Z = griddata((df['ALPHA'], df['EPSILON']), df['Energy Residual'], (X, Y), method='cubic')
# Plot the 3D surface
surf = ax.plot_surface(X, Y, Z, cmap='viridis', alpha=0.7, edgecolor='k', linewidth=0.5)

# Add labels and title
ax.set_xlabel(r'$Alpha\,{α}   $'        , fontsize=10, rotation=30)
ax.set_ylabel(r'$Epsilon\,{ϵ}   $'      , fontsize=10, rotation=0)
ax.set_zlabel(r'$Energy\, (J)     $'    , fontsize=10, rotation=0)


formatter_major = ScalarFormatter(useMathText=True)
formatter_major.set_powerlimits((-3, 2))  # Adjust power limits if needed
ax.zaxis.set_major_formatter(formatter_major)

ax.zaxis.set_major_locator(MultipleLocator(40000))
ax.zaxis.set_minor_locator(MultipleLocator(10000))

ax.yaxis.set_major_locator(MultipleLocator(0.2))
ax.yaxis.set_major_formatter('{x:.2f}')
ax.yaxis.set_minor_locator(MultipleLocator(0.05))
ax.xaxis.set_major_locator(MultipleLocator(0.2))
ax.xaxis.set_major_formatter('{x:.2f}')
ax.xaxis.set_minor_locator(MultipleLocator(0.05))

ax.xaxis.set_tick_params(labelsize=8)
ax.yaxis.set_tick_params(labelsize=8)
ax.zaxis.set_tick_params(labelsize=8)

# Show the plot

max_index = np.unravel_index(np.nanargmax(Z, axis=None), Z.shape)
min_index = np.unravel_index(np.nanargmin(Z, axis=None), Z.shape)
avg_Z = np.nanmean(Z)
avg_index = np.unravel_index(np.nanargmin(np.abs(Z - avg_Z)), Z.shape)

# Extract corresponding X, Y, and Z values
max_X = X[max_index]
max_Y = Y[max_index]
max_Z = Z[max_index]

min_X = X[min_index]
min_Y = Y[min_index]
min_Z = Z[min_index]

avg_X = X[avg_index]
avg_Y = Y[avg_index]
avg_Z = Z[avg_index]

print(f"\n\nMaximum value of Energy Residual : {max_Z} | alpha : {max_X} | epsilon : {max_Y} \nMinimum value of Energy Residual : {min_Z} | alpha : {min_X} | epsilon : {min_Y} \nAverage value of Energy Residual : {avg_Z} | alpha : {avg_X} | epsilon : {avg_Y} \n\n")
plt.show()