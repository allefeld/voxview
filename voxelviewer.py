#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# voxel viewer aka "brain game"
# a new beginning
# CA 2019-6–6


from dataclasses import dataclass
import numpy as np
import matplotlib.colors as mc


# volumes used for display
@dataclass
class Volume:
    data:   np.array = np.array([[[0]]])
    affine: np.array = np.eye(4)

    def __eq__(self, volume):
        # test for equality of Volume objects
        #   This is necessary because the __eq__ method automatically generated
        # by dataclass delegates comparison to np.array, which returns an array
        # of element-wise comparisons. Moreover, it is necessary to use
        # np.testing.assert_equal because np.array_equal treats NaN as
        # nonidentical.
        try:
            np.testing.assert_equal(self.data, volume.data)
            np.testing.assert_equal(self.affine, volume.affine)
        except AssertionError:
            return False
        return True


# surfaces to be displayed
@dataclass
class Surface:
    volumeID:  int          # volume to be thresholded
    threshold: float        # threshold value
    ka:        np.array     # Phong ambient
    kd:        np.array     # Phong diffuse
    ks:        np.array     # Phong specular
    alpha:     float        # Phong shininess


class VoxelViewer:
    def __init__(self):
        self.volumes = []
        self.surfaces = []

    def addVolume(self, volumeSpec, scan=0):
        """add volume to the list of volumes

        ``volumeSpec`` can be

        – a 2-tuple containing a 3d numpy array (voxel data) and a 4×4 numpy
        array (affine transformation from augmented voxel indices into world
        coordinates),

        – or the name of a file that can be read by nibabel.

        The data should be 3- or 4-dimensional. For 4d, a single ``scan`` must
        be selected.

        The return value is a volume ID starting from 0."""
        # create Volume from volumeSpec
        if type(volumeSpec) == tuple:
            data = volumeSpec[0]
            affine = volumeSpec[1]
        elif type(volumeSpec) == str:
            import nibabel  # only introduce dependency if functionality is used
            img = nibabel.load(volumeSpec)
            data = img.get_fdata()
            affine = img.affine
        else:
            raise TypeError
        if (data.ndim < 3) or (data.ndim > 4):
            raise NotImplementedError("Data must be 3d or 4d.")
        if data.ndim == 4:
            data = data[:, :, :, scan]
        volume = Volume(data, affine)
        # add it to the volume list and obtain index
        # or obtain index of existing idenical element
        print(self.volumes.index(volume))
        if volume not in self.volumes:
            volumeID = len(self.volumes)
            self.volumes.append(volume)
        else:
            volumeID = self.volumes.index(volume)
        # return the index as the volume ID
        return volumeID

    def addSurface(self, volumeID, threshold, colorSpec):
        """add surface to the list of surfaces

        A surface is defined by applying a ``threshold`` to a volume
        (``volumeID``). Its color is specified using a matplotlib ``colorSpec``,
        see https://matplotlib.org/3.1.0/tutorials/colors/colors.html
        """
        # obtain RGB color from colorSpec
        color = mc.to_rgb(colorSpec)
        # calculate Phong illumination coefficients from color
        ka = np.array(color) * 0.3
        kd = np.array(color) * 0.3
        ks = np.array([1., 1., 1.]) * 0.1
        alpha = 10.
        # create Surface and add it to the list
        surface = Surface(volumeID, threshold, ka, kd, ks, alpha)
        self.surfaces.append(surface)


vv = VoxelViewer()
vol1 = vv.addVolume("sLPcomb-radek-X-C-PxC.nii")
vol2 = vv.addVolume((np.array([[[1]]]), np.eye(4)))
vol3 = vv.addVolume("sLPcomb-radek-X-C-PxC.nii")
vv.addSurface(vol1, 0.007, 'r')
vv.addSurface(vol2, 0.5, (0, 0, 1))

