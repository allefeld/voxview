#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from voxelviewer import VoxelViewer

# example user program
vv = VoxelViewer()

# vol = vv.addVolume("sLPcomb-radek-X-C-PxC.nii")
# vv.addSurface(vol, 0.0174, 'b')
# vv.addSurface(vol, 0.007, 'r')

import numpy as np
x = np.array([
    [
        [np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan],
        [np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan],
        [np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan],
        [np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan],
        [np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan]
    ],
    [
        [np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan],
        [np.nan, 0, 1, 2, 3, 4, 5, np.nan],
        [np.nan, 0, 1, 2, 3, 4, 5, np.nan],
        [np.nan, 0, 1, 2, 3, 4, 5, np.nan],
        [np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan]
    ],
    [
        [np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan],
        [np.nan, 0, 1, 2, 3, 4, 5, np.nan],
        [np.nan, 0, 1, 2, 3, 4, 5, np.nan],
        [np.nan, 0, 1, 2, 3, 4, 5, np.nan],
        [np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan]
    ],
    [
        [np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan],
        [np.nan, 0, 1, 2, 3, 4, 5, np.nan],
        [np.nan, 0, 1, 2, 3, 4, 5, np.nan],
        [np.nan, 0, 1, 2, 3, 4, 5, np.nan],
        [np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan]
    ],
    [
        [np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan],
        [np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan],
        [np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan],
        [np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan],
        [np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan]
    ]
])
vol = vv.addVolume((x, 10. * np.eye(4)), nanValue=0)
vv.addSurface(vol, 2.5, 'y')

# import numpy as np
# x = np.random.rand(20, 20, 20)
# vol = vv.addVolume((x, 10. * np.eye(4)))
# vv.addSurface(vol, 0.6, 'y')
# vol = vv.addVolume((1. - x, 10. * np.eye(4)))
# vv.addSurface(vol, 0.6, 'r')

# vol = vv.addVolume("/home/ca/Store/lab/dicom/nifti/07 T1w_MPR GR_IR.nii")
# vv.addSurface(vol, 4096., 'r')
# vol = vv.addVolume("/home/ca/Store/lab/dicom/nifti/09 T2w_SPC SE.nii")
# vv.addSurface(vol, 2745., 'g')
