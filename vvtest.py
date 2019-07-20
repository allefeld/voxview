#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from voxelviewer import VoxelViewer
import numpy as np

# example user program
vv = VoxelViewer()

vol = vv.addVolume("mni_icbm152_nlin_asym_09c"
                   "/mni_icbm152_gm_tal_nlin_asym_09c.nii")
vv.addSurface(vol, 0.333, [0.5, 0.5, 0.5])
vol = vv.addVolume("sLPcomb-radek-X-C-PxC.nii")
vv.addVoxels(vol, 0.007, 'r')

# x = np.array([
#     [
#         [np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan],
#         [np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan],
#         [np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan],
#         [np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan],
#         [np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan]
#     ],
#     [
#         [np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan],
#         [np.nan, 0, 1, 2, 3, 4, 5, np.nan],
#         [np.nan, 0, 1, 2, 3, 4, 5, np.nan],
#         [np.nan, 0, 1, 2, 3, 4, 5, np.nan],
#         [np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan]
#     ],
#     [
#         [np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan],
#         [np.nan, 0, 1, 2, 3, 4, 5, np.nan],
#         [np.nan, 0, 1, 2, 3, 4, 5, np.nan],
#         [np.nan, 0, 1, 2, 3, 4, 5, np.nan],
#         [np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan]
#     ],
#     [
#         [np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan],
#         [np.nan, 0, 1, 2, 3, 4, 5, np.nan],
#         [np.nan, 0, 1, 2, 3, 4, 5, np.nan],
#         [np.nan, 0, 1, 2, 3, 4, 5, np.nan],
#         [np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan]
#     ],
#     [
#         [np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan],
#         [np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan],
#         [np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan],
#         [np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan],
#         [np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan]
#     ]
# ])
# vol = vv.addVolume((x, 10. * np.eye(4)), nanValue=0)
# vv.addSurface(vol, 2.5, 'y')

# x = np.random.rand(20, 20, 20)
# vol = vv.addVolume((x, 10. * np.eye(4)))
# vv.addSurface(vol, 0.6, 'y')
# vol = vv.addVolume((1. - x, 10. * np.eye(4)))
# vv.addSurface(vol, 0.6, 'r')

# vol = vv.addVolume("/home/ca/Store/lab/dicom/nifti/07 T1w_MPR GR_IR.nii")
# vv.addSurface(vol, 4096., 'r')
# vol = vv.addVolume("/home/ca/Store/lab/dicom/nifti/09 T2w_SPC SE.nii")
# vv.addSurface(vol, 2745., 'g')

# x = np.array([
#     [
#         [1, 1, 1],
#         [1, 0, 0],
#         [1, 0, 0]
#     ],
#     [
#         [1, 0, 0],
#         [0, 0, 0],
#         [0, 0, 0]
#     ],
#     [
#         [1, 0, 0],
#         [0, 0, 0],
#         [0, 0, 0]
#     ]
# ])
# vol = vv.addVolume(x)
# vv.addSurface(vol, 0.95, 'k')
# y = np.array([[[1]]])
# vol = vv.addVolume(y)
# vv.addSurface(vol, 0.5, 'w')
# ax = np.array([[1, 0, 0, 2],
#                [0, 1, 0, 0],
#                [0, 0, 1, 0],
#                [0, 0, 0, 1]])
# ay = np.array([[1, 0, 0, 0],
#                [0, 1, 0, 2],
#                [0, 0, 1, 0],
#                [0, 0, 0, 1]])
# az = np.array([[1, 0, 0, 0],
#                [0, 1, 0, 0],
#                [0, 0, 1, 2],
#                [0, 0, 0, 1]])
# vol = vv.addVolume((y, ax))
# vv.addSurface(vol, 0.5, 'r')
# vol = vv.addVolume((y, ay))
# vv.addSurface(vol, 0.5, 'g')
# vol = vv.addVolume((y, az))
# vv.addSurface(vol, 0.5, 'b')
