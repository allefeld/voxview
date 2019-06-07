#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
voxel viewer aka "brain game"

a new beginning

CA 2019-6–6
"""


from dataclasses import dataclass
import numpy as np
import matplotlib.colors as mc
import sdl2
import OpenGL.GL as gl


# volumes used for display
@dataclass
class Volume:
    data:   np.array = np.array([[[0]]])
    affine: np.array = np.eye(4)

    def __eq__(self, volume):
        """
        test for equality of Volume objects

        This is necessary because the __eq__ method automatically generated
        by dataclass delegates comparison to np.array, which returns an array
        of element-wise comparisons. Moreover, it is necessary to use
        np.testing.assert_equal because np.array_equal treats NaN as
        nonidentical.
        """
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
    ka:        np.array     # Phong ambient RGB
    kd:        np.array     # Phong diffuse RGB
    ks:        np.array     # Phong specular RGB
    alpha:     float        # Phong shininess


class VoxelViewer:

    # environment --------------------------------------------------------------
    #   This includes SDL2 (with game controller) and OpenGL context

    def initEnvironment(self):
        # initialize SDL
        sdl2.SDL_Init(sdl2.SDL_INIT_VIDEO | sdl2.SDL_INIT_GAMECONTROLLER)

        # look for game controller
        self.gc = None
        for i in range(sdl2.SDL_NumJoysticks()):
            if sdl2.SDL_IsGameController(i) == sdl2.SDL_TRUE:
                self.gc = sdl2.SDL_GameControllerOpen(i)
                break
        if self.gc is None:
            raise RuntimeError("No game controller found!")
        # create window & hide mouse
        window = sdl2.SDL_CreateWindow(
            "shader".encode('UTF-8'),
            sdl2.SDL_WINDOWPOS_UNDEFINED, sdl2.SDL_WINDOWPOS_UNDEFINED,
            800, 600,
            sdl2.SDL_WINDOW_OPENGL | sdl2.SDL_WINDOW_RESIZABLE)
        sdl2.SDL_ShowCursor(sdl2.SDL_DISABLE)
        # create OpenGL context
        self.context = sdl2.SDL_GL_CreateContext(window)

    def initProgram(self):
        """
        init shader program

        creates vertex data, creates and compiles vertex and fragment shader,
        creates program object, and transfers vertex data
        TODO: transfer volume and surface data
        """
        # based on http://www.hivestream.de/python-3-and-opengl-woes.html

        # create and activate Vertex Array Object (VAO)
        VAO = gl.glGenVertexArrays(1)
        gl.glBindVertexArray(VAO)

        # define vertex data describing a quad
        #   The quad is coded as two triangles with two shared vertices,
        # 1-2-3 & 2-3-4, a "triangle strip":
        #   3-4
        #   |\|
        #   1-2
        quad = np.array([-1, -1, 1, -1, -1, 1, 1, 1], dtype=np.float32)

        # create a buffer object intended for the vertex data,
        # therefore called vertex buffer object (VBO)
        VBO = gl.glGenBuffers(1)
        # initialize the VBO to be used for vertex data
        gl.glBindBuffer(gl.GL_ARRAY_BUFFER, VBO)
        # create mutable storage for the VBO and copy data into it
        gl.glBufferData(gl.GL_ARRAY_BUFFER, quad, gl.GL_STATIC_DRAW)

        # create and compile vertex shader
        #   This vertex shader simply puts vertices at our 2D positions.
        vertexShaderSource = """#version 100
            in vec2 position;
            void main() {
                gl_Position = vec4(position, 0., 1.);
            }"""
        vertexShader = gl.glCreateShader(gl.GL_VERTEX_SHADER)
        gl.glShaderSource(vertexShader, vertexShaderSource)
        gl.glCompileShader(vertexShader)
        if gl.glGetShaderiv(vertexShader, gl.GL_COMPILE_STATUS) == gl.GL_TRUE:
            print("*** OpenGL vertex shader compiled.")
        else:
            raise RuntimeError(
                "OpenGL vertex shader could not be compiled\n"
                + gl.glGetShaderInfoLog(vertexShader).decode('ASCII'))

        # create and compile fragment shader
        with open("voxelviewer.glsl", 'r') as file:
            fragmentShaderSource = file.read()
        fragmentShaderSource = fragmentShaderSource \
            .replace("#define NV 0", "#define NV %d" % len(self.volumes))\
            .replace("#define NS 0", "#define NS %d" % len(self.surfaces))
        fragmentShader = gl.glCreateShader(gl.GL_FRAGMENT_SHADER)
        gl.glShaderSource(fragmentShader, fragmentShaderSource)
        gl.glCompileShader(fragmentShader)
        if gl.glGetShaderiv(fragmentShader, gl.GL_COMPILE_STATUS) == gl.GL_TRUE:
            print("*** OpenGL fragment shader compiled.")
        else:
            raise RuntimeError(
                "OpenGL fragment shader could not be compiled\n"
                + gl.glGetShaderInfoLog(fragmentShader).decode('ASCII'))

        # create program object and attach shaders
        program = gl.glCreateProgram()
        gl.glAttachShader(program, vertexShader)
        gl.glAttachShader(program, fragmentShader)
        # make name of fragment shader color output explicit
        gl.glBindFragDataLocation(program, 0, b"fragColor")
        # link the program
        gl.glLinkProgram(program)
        if gl.glGetProgramiv(program, gl.GL_LINK_STATUS) == gl.GL_TRUE:
            print("*** OpenGL program linked.")
        else:
            raise RuntimeError(
                "OpenGL program could not be linked\n"
                + gl.glGetProgramInfoLog(program).decode('ASCII'))
        # validate the program
        gl.glValidateProgram(program)
        # activate the program
        gl.glUseProgram(program)

        # specify the layout of our vertex data
        #   get a handle for the input variable position in our shader program
        posAttrib = gl.glGetAttribLocation(program, b"position")
        #   activate this input
        gl.glEnableVertexAttribArray(posAttrib)
        #   format of the vertex data
        # Here it is defined as consisting of pairs of GL_FLOAT type items
        # with no other items between them (stride parameter 0) starting at
        # offset 0 in the buffer. This function refers to the currently bound
        # GL_ARRAY_BUFFER, which is our vbo with the corner coordinates of
        # the quad.
        gl.glVertexAttribPointer(posAttrib, 2, gl.GL_FLOAT, False, 0,
                                 gl.GLvoidp(0))

    # shader state -------------------------------------------------------------
    #   This includes everything whose change makes it necessary to generate a
    # new shader program.

    def initShaderState(self):
        self.volumes = []
        self.surfaces = []

    def addVolume(self, volumeSpec, scan=0):
        """
        add volume to the list of volumes

        ``volumeSpec`` can be

        – a 2-tuple containing a 3d numpy array (voxel data) and a 4×4 numpy
        array (affine transformation from augmented voxel indices into world
        coordinates),

        – or the name of a file that can be read by nibabel.

        The data should be 3- or 4-dimensional. For 4d, a single ``scan`` must
        be selected.

        The return value is a volume ID starting from 0.
        """
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
        # or obtain index of existing identical element
        if volume not in self.volumes:
            volumeID = len(self.volumes)
            self.volumes.append(volume)
        else:
            volumeID = self.volumes.index(volume)
        # return the index as the volume ID
        return volumeID

    def addSurface(self, volumeID, threshold, colorSpec):
        """
        add surface to the list of surfaces

        A surface is defined by applying a ``threshold`` to a volume
        (``volumeID`` returned by addVolume).

        Its color is specified using a matplotlib ``colorSpec``,
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

    # --------------------------------------------------------------------------

    def __init__(self):
        # init environment
        self.gc = self.context = None
        self.initEnvironment()
        # init shader state
        self.volumes = self.surfaces = None
        self.initShaderState()

    def run(self):
        # init shader program
        self.initProgram()


# user program
vv = VoxelViewer()
vol1 = vv.addVolume("sLPcomb-radek-X-C-PxC.nii")
vol2 = vv.addVolume((np.array([[[1]]]), np.eye(4)))
vol3 = vv.addVolume("sLPcomb-radek-X-C-PxC.nii")
vv.addSurface(vol1, 0.007, 'r')
vv.addSurface(vol2, 0.5, (0, 0, 1))
vv.run()
