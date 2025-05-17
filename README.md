# voxview

Navigate through a 3D view of voxel volumes using a game controller.

> [!NOTE]
> Working but unfinished project.

`vvtest.py`: Demo Python script, get started by running it. Uses class `VoxelViewer`.\
`voxelviewer.py`: Python module which defines the class `VoxelViewer` which can be used to construct visualizations of 3d volumes defined by voxel arrays.\
`voxelviewer.glsl`: OpenGL shader used by `VoxelViewer`.
`mni_icbm152_gm_tal_nlin_asym_09c.nii`, `sLPcomb-radek-X-C-PxC.nii`: NIfTI files used by the demo script.

(Some) needed packages: `pyopengl`, `pysdl2`, `matplotlib`, `numpy`.

***

This software is copyrighted © 2019–2025 by Carsten Allefeld and released under the terms of the GNU General Public License, version 3 or later.