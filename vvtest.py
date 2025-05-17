#!/usr/bin/env python3

from voxelviewer import VoxelViewer

# example user program
vv = VoxelViewer()

vol = vv.addVolume("mni_icbm152_gm_tal_nlin_asym_09c.nii")
vv.addSurface(vol, 0.333, [0.5, 0.5, 0.5])
vol = vv.addVolume("sLPcomb-radek-X-C-PxC.nii")
vv.addVoxels(vol, 0.007, 'r')
