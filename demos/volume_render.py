#!/usr/bin/env python
import time

import numpy as np
import icecream

import sci3d as s3d

icecream.install()

count = 128j

# cube
buff = np.mgrid[0:1:count, 0:1:count, 0:1:count].astype(np.float32)
cube = np.maximum(
    np.maximum(0.5 - buff[0, ...], -0.5 + buff[0, ...]),
    np.maximum(np.maximum(0.5 - buff[1, ...], -0.5 + buff[1, ...]),
               np.maximum(0.5 - buff[2, ...], -0.5 + buff[2, ...]))
) - 0.25

# sphere
buff = np.mgrid[-1:1:count, -1:1:count, -1:1:count].astype(np.float32)
sphere = np.linalg.norm(buff, 2, 0) - 0.5

"""
# armadillo
sdf = np.load("/home/claudio/workspace/adventures-in-tensorflow/volume_armadillo.npz")
sdf = (sdf['scalar_field'] - sdf['target_level']).astype(np.float32)
# Invert SDF sign if it is negative outside
sdf = sdf * np.sign(sdf[0, 0, 0])
# """

sdf = cube

s1 = s3d.isosurface(cube, title='cube')
s2 = s3d.isosurface(sphere, title='sphere')

cube_light_pos = np.array([
    [0, 0, 0],
    [1, 0, 0],
], dtype=np.float32)
cube_light_color = np.array([
    [1, 1, 1],
    [1, 0, 0],
], dtype=np.float32)
s1.set_lights(cube_light_pos, cube_light_color)

t = 0
dt = 1.0/60.0
while s3d.get_window_count() > 0:
    alpha = np.cos(t) * 0.5 + 0.5
    sdf = alpha * cube + (1-alpha) * sphere
    s2.set_isosurface(sdf)
    time.sleep(dt)
    t += dt * 3

s3d.shutdown()
