#!/usr/bin/env python
import time

import numpy as np
import icecream

import sci3d as s3d

icecream.install()

ic()
# cube
buff = np.mgrid[0:1:196j, 0:1:196j, 0:1:196j].astype(np.float32)
sdf = np.maximum(
    np.maximum(0.5 - buff[0, ...], -0.5 + buff[0, ...]),
    np.maximum(np.maximum(0.5 - buff[1, ...], -0.5 + buff[1, ...]),
               np.maximum(0.5 - buff[2, ...], -0.5 + buff[2, ...]))
) - 0.25
"""
# sphere
buff = np.mgrid[-1:1:128j, -1:1:128j, -1:1:128j].astype(np.float32)
sdf = np.linalg.norm(buff, 2, 0) - 0.5
"""
"""
# armadillo
sdf = np.load("/home/claudio/workspace/adventures-in-tensorflow/volume_armadillo.npz")
sdf = (sdf['scalar_field'] - sdf['target_level']).astype(np.float32)
# """

# Invert SDF sign if it is negative outside
sdf = sdf * np.sign(sdf[0, 0, 0])

ic('call isosurf...')
s3d.isosurface(sdf)
ic('calling isosurf 2')
s3d.isosurface(sdf)
ic('second call done')
while s3d.are_windows_open():
    time.sleep(0.5)

s3d.shutdown()
