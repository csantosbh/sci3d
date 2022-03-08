#!/usr/bin/env python
import time

import numpy as np
import sci3d as s3d


# Make mesh
vertices = np.array([
    [-0.25, -0.25, 0.25],  # 0
    [0.25, -0.25, 0.25],   # 1
    [0.25, 0.25, 0.25],    # 2
    [-0.25, 0.25, 0.25],   # 3

    [-0.25, -0.25, -0.25],  # 4
    [0.25, -0.25, -0.25],   # 5
    [0.25, 0.25, -0.25],    # 6
    [-0.25, 0.25, -0.25],   # 7
], dtype=np.float32)

indices = np.array([
    # Frontal faces
    [0, 1, 2],
    [2, 3, 0],
    [4, 5, 6],
    [6, 7, 4],
    [0, 3, 7],
    [7, 4, 0],

    # Back faces
    [1, 0, 2],
    [3, 2, 0],
    [5, 4, 6],
    [7, 6, 4],
    [3, 0, 7],
    [4, 7, 0],
], dtype=np.uint32)

# Create mesh. Note the usage of the optional position/rotation fields
fig = s3d.mesh(vertices, indices, common_params=s3d.Params(
    object_position=np.array([[0, 0, -1]], dtype=np.float32).T,
    object_rotation=s3d.look_at(
        np.array([[0.5, 0, 0.5]], dtype=np.float32).T,
        np.array([[0, 1, 0]], dtype=np.float32).T,
    ),
))

# Time tracking
t = 0
dt = 1.0/60.0

while s3d.get_window_count() > 0:
    # Update mesh vertices
    fig.set_mesh(
        vertices=(
            vertices + np.cos(t + vertices[:, 0:1] * 4) *
            np.array([[0, 0.1, 0]], dtype=np.float32)
        ),
        triangles=indices
    )

    # Change mesh transform
    fig.set_transform(
        np.array([[0, 0, 0]], dtype=np.float32).T,
        s3d.look_at(np.array([[np.cos(t), 0, np.sin(t)]], dtype=np.float32).T,
                    np.array([[0, 1, 0]], dtype=np.float32).T)
    )

    time.sleep(dt)
    t += dt * 3

# Always shutdown sci3d after you're done!
s3d.shutdown()
