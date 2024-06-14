from scipy.stats.qmc import LatinHypercube as LHSampler
from matplotlib import pyplot as plt
import numpy as np
from math import floor
import random as r
from decimal import Decimal

# 2D-only LHS plotter
def plotLHS(lhs: np.ndarray, grid: bool = False, highlight: bool = False, save: bool = False, filepath: str = None):
    N, P = lhs.shape
    if P != 2:
        return
    plt.xlim(0, 1)
    plt.ylim(0, 1)
    plt.scatter(lhs[:, 0], lhs[:, 1], marker='o', c='r', s=2)
    if grid:
        for q in range(0, N):
            plt.axhline(y=q/N, color='black', linestyle='--', linewidth=0.3)
            plt.axvline(x=q/N, color='black', linestyle='--', linewidth=0.3)
    # floor(coord/interval_size) = interval_index (starting from zero)
    if highlight:
        timestep = 1/N
        for i in range(N):
            try:
                if lhs[i, 0] is None or lhs[i, 0] is None or np.isnan(lhs[i, 0]) or np.isnan(lhs[i, 1]):
                    continue
                qh = floor(lhs[i, 0]/timestep)
                qv = floor(lhs[i, 1]/timestep)
                plt.axvspan(qh/N, (qh+1)/N, facecolor='blue', alpha=0.15)
                plt.axhspan(qv/N, (qv+1)/N, facecolor='blue', alpha=0.15)
            except:
                continue
    if save and filepath is not None:
        plt.savefig(filepath)
    else:
        plt.show()




# other utilities 
def concat(a: np.ndarray, b:np.ndarray):
    return np.concatenate((a, b), axis=0)

def inner_coords(point, N):
    timestep = 1/N
    # increases rounding error
    return [ point[j] * N - floor(point[j]/timestep) for j in range(len(point))]

def rpop(v: np.ndarray):
    rindex = r.randint(0, len(v) - 1)
    return v[rindex], np.delete(v, rindex, 0)

# Approx Heaviside step function
def F(t, sharpness = 1000):
    return 0.5 * (1 + np.tanh(sharpness * t))

# Approx Heaviside step function
def H(x):
    return np.where(x >= 0, 1, 0)

# high precision difference
def high_precision_difference(a: float, b: float, rounding_error = 30):
    return round(
        float(Decimal(str(a)) - Decimal(str(b))), 
        rounding_error
    )