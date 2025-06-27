#!/usr/bin/env python

import numpy as np
import matplotlib.pyplot as plt

def draw_box(ax, box, style='r-', fill=True):
    ax.plot([box[0], box[2]], [box[1], box[1]], style) # Bottom
    ax.plot([box[0], box[2]], [box[3], box[3]], style) # Top
    ax.plot([box[0], box[0]], [box[1], box[3]], style) # Left
    ax.plot([box[2], box[2]], [box[1], box[3]], style) # Right

def f(d):
    B = np.pi / 2.0
    A = - np.pi / RAMP_WIDTH
    return 0.5 * (1.0 + np.sin(A * d + B))

def ramp(x, y, window):
    # Distance to closest edge
    d = np.min(np.array([x - window[0], 
                         window[2] - x, 
                         y - window[1], 
                         window[3] - y]), axis=0)
    return (-RAMP_WIDTH < d) * (d < 0.0) * f(d) + (d > 0) * np.ones(x.shape) + (d < -RAMP_WIDTH) * np.zeros(x.shape)

RAMP_WIDTH = 1.25
P_ambient = 1013
P_center = 900
domain = np.array([-85.0, 20.0, -60.0, 50.0])
window = np.array([-75, 30, -65, 40])
deg_factor = 8
x = np.linspace(domain[0], domain[2], int(domain[2] - domain[0]) * deg_factor)
y = np.linspace(domain[1], domain[3], int(domain[3] - domain[1]) * deg_factor)
X, Y = np.meshgrid(x, y)

fig, ax = plt.subplots()
# plot = ax.pcolormesh(X, Y, P_ambient + (P_center - P_ambient) * ramp(X, Y, ramp_window)[:-1, :-1], shading='flat')
plot = ax.pcolormesh(X, Y, ramp(X, Y, window)[:-1, :-1], shading='flat')
draw_box(ax, window)
fig.colorbar(plot)
ax.set_xlabel("longitude")
ax.set_ylabel("latitude")
plt.show()