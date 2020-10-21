import pandas as pd
import numpy as np
import os
from glob import glob
import matplotlib.pyplot as plt
from matplotlib.patches import Circle, RegularPolygon
from matplotlib.path import Path
from matplotlib.projections.polar import PolarAxes
from matplotlib.projections import register_projection
from matplotlib.spines import Spine
from matplotlib.transforms import Affine2D 
from matplotlib.patches import Ellipse
import seaborn as sns
from math import pi


def draw_ellipse(position, covariance, ax=None, **kwargs):
    """Draw an ellipse with a given position and covariance"""
    ax = ax or plt.gca()
    # Convert covariance to principal axes
    if covariance.shape == (2, 2):
        U, s, Vt = np.linalg.svd(covariance)
        angle = np.degrees(np.arctan2(U[1, 0], U[0, 0]))
        width, height = 2 * np.sqrt(s)
    else:
        angle = 0
        width, height = 2 * np.sqrt(covariance)
    for nsig in range(1, 4):
        ax.add_patch(Ellipse(position, nsig * width, nsig * height,
                             angle, **kwargs))



def select_best(arr:list, X:int)->list:
    '''
    returns the set of X configurations with shorter distance
    '''
    dx=np.argsort(arr)[:X]
    return arr[dx]

def plot_gmm(gmm, X, label=True, ax=None):
    ax = ax or plt.gca()
    clusters = gmm.fit(X).predict(X)
    if label:
        for cluster_num in range(gmm.n_components):
            sns.scatterplot(
                x=X[clusters == cluster_num, 0], y=X[clusters == cluster_num, 1],  s=40,  zorder=2,  label='$cluster %i$'%cluster_num

            )    
        # ax.scatter(X[:, 0], X[:, 1], c=labels, s=40, cmap='viridis', zorder=2,  label=labels)
    else:
        ax.scatter(X[:, 0], X[:, 1], s=40, zorder=2)
    
    w_factor = 0.2 / gmm.weights_.max()
    for pos, covar, w in zip(gmm.means_, gmm.covariances_, gmm.weights_):
        draw_ellipse(pos, covar, alpha=w * w_factor)
    plt.title("HEEEEELLL!!!! %d components"%len(gmm.means_), fontsize=(20))
    plt.legend(loc='upper center', bbox_to_anchor=(0.5, 1.20),
          ncol=3, fancybox=True, shadow=True)
    plt.xlabel("U.A.")
    plt.ylabel("U.A.")






