#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from voxelviewer import VoxelViewer
import numpy as np

# example user program
vv = VoxelViewer()

# vol = vv.addVolume("sLPcomb-radek-X-C-PxC.nii")
# vv.addSurface(vol, 0.0174, 'b')
# vv.addSurface(vol, 0.007, 'r')

x = np.random.rand(20, 20, 20)
vol = vv.addVolume((x, 10. * np.eye(4)))
vv.addSurface(vol, 0.6, 'y')
vol = vv.addVolume((1. - x, 10. * np.eye(4)))
vv.addSurface(vol, 0.6, 'b')
