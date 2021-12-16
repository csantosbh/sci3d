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

# Make mesh
vertices = np.array([
    [0.25, 0.25, -0.25],  # 0
    [0.75, 0.25, -0.25],  # 1
    [0.75, 0.75, -0.25],  # 2
    [0.25, 0.75, -0.25],  # 3

    [0.25, 0.25, -0.75],  # 4
    [0.75, 0.25, -0.75],  # 5
    [0.75, 0.75, -0.75],  # 6
    [0.25, 0.75, -0.75],  # 7
], dtype=np.float32)
indices = np.array([
    [0, 1, 2],
    [2, 3, 0],
    [4, 5, 6],
    [6, 7, 4],
    [0, 3, 7],
    [7, 4, 0],



    [1, 0, 2],
    [3, 2, 0],
    [5, 4, 6],
    [7, 6, 4],
    [3, 0, 7],
    [4, 7, 0],
], dtype=np.uint32)

s2 = s3d.mesh(vertices, indices)
s1 = s3d.isosurface(cube, title='cube')

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
    #s1.set_lights(cube_light_pos, cube_light_color)
    #s2.set_mesh(vertices=(vertices + np.array([[np.cos(t*0.2)/10, 0, 0]], dtype=np.float32)))

    time.sleep(dt)
    t += dt * 3


s3d.shutdown()
