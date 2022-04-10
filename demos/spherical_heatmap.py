#!/usr/bin/env python
import time

import numpy as np
import sci3d as s3d
import icecream
icecream.install()


# Make mesh
points = np.array([
    [-1, -1, -1],
    [-1, -1, 1],
    [1,  -1, 1],
    [1,  -1, -1],
    [-1, 1, -1],
    [-1, 1, 1],
    [1, 1, 1],
    [1, 1, -1],
], dtype=np.float32)

points = np.random.normal(size=(100, 3)).astype(np.float32) + np.array(
    [[0, 0, 1]], dtype=np.float32)
points = points / np.linalg.norm(points, axis=1, keepdims=True)

# Create mesh. Note the usage of the optional position/rotation fields
fig = s3d.spherical_heatmap(points)
lpos = np.array([
    [0, 0, 2],
    [2, 0, 0],
    [0, 2, 1],
    [0, -2, -1],
], dtype=np.float32)
lcol = np.array([
    [1, 1, 1],
    [1, 1, 1],
    [0, 0, 0],
    [0, 0, 0],
], dtype=np.float32)
fig.set_lights(lpos, lcol)

# Time tracking
t = 0
dt = 1.0/60.0

while s3d.get_window_count() > 0:
    # Change mesh transform
    fig.set_transform(
        np.array([[0, 0, 0]], dtype=np.float32).T,
        s3d.look_at(np.array([[np.cos(t/10), 0, np.sin(t/10)]], dtype=np.float32).T,
                    np.array([[0, 1, 0]], dtype=np.float32).T)
    )
    fig.set_points(points + np.array([[0, 0, np.cos(t*10)]], dtype=np.float32))

    time.sleep(dt)
    t += dt

# Always shutdown sci3d after you're done!
s3d.shutdown()
