#!/usr/bin/env python
import time

import numpy as np

import sci3d
import sci3d as s3d


resolution = 128j

# Make cube
buff = np.mgrid[0:1:resolution, 0:1:resolution, 0:1:resolution].astype(np.float32)
cube = np.maximum(
    np.maximum(0.5 - buff[0, ...], -0.5 + buff[0, ...]),
    np.maximum(np.maximum(0.5 - buff[1, ...], -0.5 + buff[1, ...]),
               np.maximum(0.5 - buff[2, ...], -0.5 + buff[2, ...]))
) - 0.25
tst = np.load('../../adventures-in-tensorflow/armadillo_volume.npz')
cube = -(tst['scalar_field'] - tst['target_level']).astype(np.float32)

# Make sphere
#buff = np.mgrid[-1:1:resolution, -1:1:resolution, -1:1:resolution].astype(np.float32)
#sphere = np.linalg.norm(buff, 2, 0) - 0.5

# Create windows
fig1 = s3d.isosurface(cube, s3d.Params(
    window_title='cube',
    object_rotation=sci3d.look_at(np.array(
        [[1,0,1]], dtype=np.float32).T, np.array([[0,1,0]], dtype=np.float32).T)
))
cube_light_pos = np.array([
   [-1.0, -1.0, -1.0],
   [1.0, 1.0, 1.0],
], dtype=np.float32)*2
cube_light_color = np.array([
   [1, .5, .5],
   [.5, 0.5, 1],
], dtype=np.float32) * 2
fig1.set_lights(cube_light_pos, cube_light_color)
#s3d.figure()
#fig2 = s3d.isosurface(sphere, s3d.Params(window_title='sphere'))

# Time tracking
t = 0
dt = 1.0/60.0

while s3d.get_window_count() > 0:
    # Update lighting on cube scene

    # Update shape on sphere scene
    alpha = np.cos(t) * 0.5 + 0.5
    #fig2.set_isosurface(alpha * cube + (1 - alpha) * sphere)

    time.sleep(dt)
    t += dt * 3

# Always shutdown sci3d after you're done!
s3d.shutdown()
