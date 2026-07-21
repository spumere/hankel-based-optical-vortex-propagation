import numpy as np
from scipy.interpolate import CubicSpline

def radial_to_cartesian(dist_1d, L, N):
    N_r = len(dist_1d)
    r = np.linspace(0, L * np.sqrt(2)/2, N_r, endpoint=True)
    x = np.linspace(-L/2, L/2, N, endpoint=True)
    y = np.linspace(-L/2, L/2, N, endpoint=True)
    X, Y = np.meshgrid(x, y)
    R = np.sqrt(X**2 + Y**2)
    cs = CubicSpline(r, dist_1d)
    dist_2d = cs(R)
    return dist_2d
