#!/usr/bin/env python
import time

import numpy as np
import icecream

import sci3d as s3d


icecream.install()

resolution = 128j

# Make cube
buff = np.mgrid[0:1:resolution, 0:1:resolution, 0:1:resolution].astype(np.float32)
cube = np.maximum(
    np.maximum(0.5 - buff[0, ...], -0.5 + buff[0, ...]),
    np.maximum(np.maximum(0.5 - buff[1, ...], -0.5 + buff[1, ...]),
               np.maximum(0.5 - buff[2, ...], -0.5 + buff[2, ...]))
) - 0.25

# Make sphere
buff = np.mgrid[-1:1:resolution, -1:1:resolution, -1:1:resolution].astype(np.float32)
sphere = np.linalg.norm(buff, 2, 0) - 0.5

s1 = s3d.isosurface(cube, s3d.Params(window_title='cube'))
s3d.figure()
s2 = s3d.isosurface(sphere, s3d.Params(window_title='sphere'))

t = 0
dt = 1.0/60.0

while s3d.get_window_count() > 0:
    # Update s1 lighting
    cube_light_pos = np.array([
        [0, 0, 0],
        [1.5, np.sin(t) * 0.5 + 0.5, 0.5],
    ], dtype=np.float32)
    cube_light_color = np.array([
        [1, 1, 1],
        [1, np.cos(2*t) * 0.5 + 0.5, np.sin(3*t) * 0.5 + 0.5],
    ], dtype=np.float32)
    s1.set_lights(cube_light_pos, cube_light_color)

    # Update s2 shape
    alpha = np.cos(t) * 0.5 + 0.5
    s2.set_isosurface(alpha * cube + (1 - alpha) * sphere)

    time.sleep(dt)
    t += dt * 3

s3d.shutdown()
