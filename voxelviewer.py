#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
voxel viewer aka "brain game"
"""


from dataclasses import dataclass
import numpy as np
import matplotlib.colors as mc
import sdl2
import OpenGL.GL as gl
import struct
import ctypes
import time
from threading import Thread


@dataclass
class Volume:
    """
    volumes used for display
    """
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

        :param Volume volume:
            volume object to compare with
        :return:
            whether the two volume objects hold the same data
        """
        try:
            np.testing.assert_equal(self.data, volume.data)
            np.testing.assert_equal(self.affine, volume.affine)
        except AssertionError:
            return False
        return True


@dataclass
class Surface:
    """
    surfaces to be displayed
    """
    volumeID:  int          # volume to be thresholded
    threshold: float        # threshold value
    ka:        np.array     # Phong ambient RGB
    kd:        np.array     # Phong diffuse RGB
    ks:        np.array     # Phong specular RGB
    alpha:     float        # Phong shininess


class VoxelViewer:

    # environment --------------------------------------------------------------
    #
    # The basic stuff that's necessary for the rest to work.
    #
    # gc:               SDL2 game controller object
    # window:           SLD2 window object
    # context:          OpenGL context
    # vertexShader:     OpenGL vertex shader object
    # fragmentShader:   OpenGL fragment shader object
    # program:          OpenGL program object
    # VAO:              OpenGL vertex array object
    # VBO:              OpenGL vertex buffer object
    # posAttrib:        OpenGL vertex shader position attribute

    def _createEnvironment(self):
        """
        create environment
        """
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
        self.window = sdl2.SDL_CreateWindow(
            b'',
            sdl2.SDL_WINDOWPOS_UNDEFINED, sdl2.SDL_WINDOWPOS_UNDEFINED,
            800, 600,
            sdl2.SDL_WINDOW_OPENGL
            | sdl2.SDL_WINDOW_RESIZABLE
            | sdl2.SDL_WINDOW_MAXIMIZED)
        sdl2.SDL_ShowCursor(sdl2.SDL_DISABLE)
        # create OpenGL context
        self.context = sdl2.SDL_GL_CreateContext(self.window)

    def _destroyEnvironment(self):
        """
        destroy environment
        :return:
        """
        sdl2.SDL_GL_DeleteContext(self.context)
        sdl2.SDL_DestroyWindow(self.window)
        sdl2.SDL_Quit()

    def _createShader(self):
        """
        create shader program

        creates and compiles vertex and fragment shader,
        creates program object,
        creates vertex data, transfers vertex data
        and transfers volume and surface data
        """
        # based on http://www.hivestream.de/python-3-and-opengl-woes.html

        # GLSL doesn't like 0-size arrays. So let's give the shader something
        # to show by default.
        vol = self.volumes
        if len(vol) == 0:
            vol = [Volume(np.array([[[1]]]), np.eye(4))]
        surf = self.surfaces
        if len(surf) == 0:
            surf = [Surface(0, 0.5,
                            np.array([1., 1., 1.]) * 0.3,
                            np.array([1., 1., 1.]) * 0.3,
                            np.array([1., 1., 1.]) * 0.1,
                            10.)]

        # create and compile vertex shader
        #   This vertex shader simply puts vertices at our 2D positions.
        vertexShaderSource = """
            #version 130
            in vec2 position;
            void main() {
                gl_Position = vec4(position, 0., 1.);
            }
            """
        self.vertexShader = gl.glCreateShader(gl.GL_VERTEX_SHADER)
        gl.glShaderSource(self.vertexShader, vertexShaderSource)
        gl.glCompileShader(self.vertexShader)
        if (gl.glGetShaderiv(self.vertexShader, gl.GL_COMPILE_STATUS)
                != gl.GL_TRUE):
            raise RuntimeError(
                "OpenGL vertex shader could not be compiled\n"
                + gl.glGetShaderInfoLog(self.vertexShader).decode('ASCII'))

        # create and compile fragment shader
        with open("voxelviewer.glsl", 'r') as file:
            fragmentShaderSource = file.read()
        fragmentShaderSource = fragmentShaderSource \
            .replace("#define NV 0", "#define NV %d" % len(vol))\
            .replace("#define NS 0", "#define NS %d" % len(surf))
        self.fragmentShader = gl.glCreateShader(gl.GL_FRAGMENT_SHADER)
        gl.glShaderSource(self.fragmentShader, fragmentShaderSource)
        gl.glCompileShader(self.fragmentShader)
        if (gl.glGetShaderiv(self.fragmentShader, gl.GL_COMPILE_STATUS)
                != gl.GL_TRUE):
            raise RuntimeError(
                "OpenGL fragment shader could not be compiled\n"
                + gl.glGetShaderInfoLog(self.fragmentShader).decode('ASCII'))

        # create program object and attach shaders
        self.program = gl.glCreateProgram()
        gl.glAttachShader(self.program, self.vertexShader)
        gl.glAttachShader(self.program, self.fragmentShader)
        # make name of fragment shader color output explicit
        gl.glBindFragDataLocation(self.program, 0, b"fragColor")
        # link the program
        gl.glLinkProgram(self.program)
        if gl.glGetProgramiv(self.program, gl.GL_LINK_STATUS) != gl.GL_TRUE:
            raise RuntimeError(
                "OpenGL program could not be linked\n"
                + gl.glGetProgramInfoLog(self.program).decode('ASCII'))
        # validate the program
        gl.glValidateProgram(self.program)
        # activate the program
        gl.glUseProgram(self.program)

        # create vertex data
        # create and activate Vertex Array Object (VAO)
        self.VAO = gl.glGenVertexArrays(1)
        gl.glBindVertexArray(self.VAO)
        # define vertex data describing a quad
        #   The quad is coded as two triangles with two shared vertices,
        # 1-2-3 & 2-3-4, a "triangle strip":
        #   3-4
        #   |\|
        #   1-2
        quad = np.array([-1, -1,
                         1, -1,
                         -1, 1,
                         1, 1], dtype=np.float32)
        # create a buffer object intended for the vertex data,
        # therefore called vertex buffer object (VBO)
        self.VBO = gl.glGenBuffers(1)
        # initialize the VBO to be used for vertex data
        gl.glBindBuffer(gl.GL_ARRAY_BUFFER, self.VBO)
        # create mutable storage for the VBO and copy data into it
        gl.glBufferData(gl.GL_ARRAY_BUFFER, quad, gl.GL_STATIC_DRAW)

        # specify the layout of our vertex data
        #   get a handle for the input variable position in our shader program
        self.posAttrib = gl.glGetAttribLocation(self.program, b"position")
        #   activate this input
        gl.glEnableVertexAttribArray(self.posAttrib)
        #   format of the vertex data
        # Here it is defined as consisting of pairs of GL_FLOAT type items
        # with no other items between them (stride parameter 0) starting at
        # offset 0 in the buffer. This function refers to the currently bound
        # GL_ARRAY_BUFFER, which is our vbo with the corner coordinates of
        # the quad.
        gl.glVertexAttribPointer(self.posAttrib, 2, gl.GL_FLOAT, False, 0,
                                 gl.GLvoidp(0))

        # transfer volume data
        for volumeID, volume in enumerate(vol):
            # get data
            data = volume.data.astype(np.float16)
            # get affine transformation
            #   rotation & scaling matrix (voxel -> world)
            AM = volume.affine[:3, :3]
            # translation (voxel -> world), world position of voxel [0, 0, 0]
            AO = volume.affine[:3, 3].astype(np.float16)
            # inverse rotation & scaling (world -> voxel)
            AiM = np.linalg.inv(AM)
            # volume array element
            v = "vol[%d]" % volumeID
            # pass volume data as texture -> uniform sampler3D
            texture = gl.glGenTextures(1)
            gl.glUniform1i(
                gl.glGetUniformLocation(self.program, v + ".data"),
                volumeID)
            gl.glActiveTexture(gl.GL_TEXTURE0 + volumeID)
            gl.glBindTexture(gl.GL_TEXTURE_3D, texture)
            gl.glTexImage3D(gl.GL_TEXTURE_3D, 0, gl.GL_R16F, *data.shape,
                            0, gl.GL_RED, gl.GL_FLOAT, data.flatten('F'))
            # should the following two be necessary when using texelFetch?!
            gl.glTexParameteri(gl.GL_TEXTURE_3D, gl.GL_TEXTURE_MAX_LEVEL, 0)
            gl.glClampColor(gl.GL_CLAMP_READ_COLOR, gl.GL_FALSE)
            gl.glEnable(gl.GL_TEXTURE_3D)
            # pass affine transformation
            gl.glUniformMatrix3fv(
                gl.glGetUniformLocation(self.program, v + ".AM"),
                1, gl.GL_FALSE, struct.pack('f' * 9, *AM.flatten('F')))
            gl.glUniform3f(
                gl.glGetUniformLocation(self.program, v + ".AO"),
                *AO.astype(np.float16))
            gl.glUniformMatrix3fv(
                gl.glGetUniformLocation(self.program, v + ".AiM"),
                1, gl.GL_FALSE, struct.pack('f' * 9, *AiM.flatten('F')))
            # Why is struct.pack necessary for glUniformMatrix3fv,
            # but not for glUniform3f?

        # transfer surface data
        for surfaceID, surface in enumerate(surf):
            # surface array element
            s = "surf[%d]" % surfaceID
            # pass volume ID
            gl.glUniform1i(
                gl.glGetUniformLocation(self.program, s + ".volID"),
                surface.volumeID)
            # pass threshold
            gl.glUniform1f(
                gl.glGetUniformLocation(self.program, s + ".threshold"),
                surface.threshold)
            # pass Phong coefficients
            gl.glUniform3f(
                gl.glGetUniformLocation(self.program, s + ".ka"),
                *surface.ka)
            gl.glUniform3f(
                gl.glGetUniformLocation(self.program, s + ".kd"),
                *surface.kd)
            gl.glUniform3f(
                gl.glGetUniformLocation(self.program, s + ".ks"),
                *surface.ks)
            gl.glUniform1f(
                gl.glGetUniformLocation(self.program, s + ".alpha"),
                surface.alpha)

        # indicate that scene state is implemented
        self.sceneChanged = False

    def _destroyShader(self):
        """
        release resources acquired by shader program creation
        """
        gl.glDisableVertexAttribArray(self.posAttrib)
        gl.glDeleteProgram(self.program)
        gl.glDeleteShader(self.fragmentShader)
        gl.glDeleteShader(self.vertexShader)
        gl.glDeleteBuffers(1, [self.VBO])
        gl.glDeleteVertexArrays(1, [self.VAO])

    def _renderFrame(self):
        """
        render frame
        """

        # update window size
        w = ctypes.c_int()
        h = ctypes.c_int()
        sdl2.SDL_GetWindowSize(self.window, w, h)
        gl.glViewport(0, 0, w, h)

        # update uniforms
        # update generic uniforms
        gl.glUniform2f(
            gl.glGetUniformLocation(self.program, "resolution"),
            float(w.value), float(h.value))
        gl.glUniform1f(
            gl.glGetUniformLocation(self.program, "time"),
            self.time)
        # update camera position uniform
        gl.glUniform3f(
            gl.glGetUniformLocation(self.program, "camPos"),
            *self.camPos)
        # update camera matrix uniform
        horizontal, vertical, center = self._camDirections()
        camMatrix = np.column_stack(
            (horizontal * self.camFovF, vertical * self.camFovF, center))
        gl.glUniformMatrix3fv(
            gl.glGetUniformLocation(self.program, "camMatrix"), 1, gl.GL_FALSE,
            struct.pack('f' * 9, *camMatrix.flatten('F')))

        # use vertex buffer data to draw
        #   The first four values of the vertex sequence are interpreted as
        # specifying a triangle strip (see above).
        gl.glClearColor(1.0, 0.5, 0.0, 1.0)
        gl.glClear(gl.GL_COLOR_BUFFER_BIT)
        gl.glDrawArrays(gl.GL_TRIANGLE_STRIP, 0, 4)

        # time display updates
        title = "VoxelViewer:  %d - %.2f - %.2f" %\
                (self.frame, self.time, self.frame / self.time)
        sdl2.SDL_SetWindowTitle(self.window, title.encode('UTF-8'))
        # These frequent title updates make the KDE panel freeze, see
        # https://bugs.kde.org/show_bug.cgi?id=365317
        # Maybe fix by showing information via overlay in window?

        # swap buffers (glFlush + glXSwapBuffers?)
        sdl2.SDL_GL_SwapWindow(self.window)
        # make timing tighter – doesn't seem to hurt framerate
        gl.glFinish()

    def _toggleFullscreen(self):
        """
        toggle between windowed and fullscreen mode
        """
        if sdl2.SDL_GetWindowFlags(self.window)\
                & sdl2.SDL_WINDOW_FULLSCREEN_DESKTOP:
            sdl2.SDL_SetWindowFullscreen(self.window, 0)
        else:
            sdl2.SDL_SetWindowFullscreen(
                self.window, sdl2.SDL_WINDOW_FULLSCREEN_DESKTOP)

    # scene state -------------------------------------------------------------
    #
    # This includes everything whose change makes it necessary to generate a
    # new shader program.
    #
    # volumes:      volumes used for display
    # surfaces:     surfaces to be displayed
    # sceneChanged: has scene changed?

    def _initializeSceneState(self):
        self.volumes = []
        self.surfaces = []
        self.sceneChanged = None

    def addVolume(self, volumeSpec, scan=0, nanValue=None):
        """
        add volume to the list of volumes

        ``volumeSpec`` can be

        – a 2-tuple containing a 3d numpy array (voxel data) and a 4 × 4 numpy
        array (affine transformation from augmented voxel indices into world
        coordinates),

        – just the 3d numpy data array, in which case the affine transformation
        is set to the 4 × 4 identity matrix

        – or the name of a file that can be read by nibabel.

        The data should be 3- or 4-dimensional. For 4d, a single ``scan`` must
        be selected.

        The affine transformation is a 4 × 4 matrix that describes
        size and shape of voxels as well as
        the origin of voxel space. In the simplest case, it has the form
        ::
            dx 0  0  ox
            0  dy 0  oy
            0  0  dz oz
            0  0  0  1
        where dx, dy, dz define the size of a voxel along the three
        dimensions, and ox, oy, oz the position of the voxel (0, 0, 0).

        :param volumeSpec:
            source of volume data
        :param int scan:
            selected volume for 4d data
        :param float nanValue:
            value to replace NaNs in the data (optional)
        :return:
            numeric ID of the volume
        """
        # create Volume from volumeSpec
        if isinstance(volumeSpec, tuple) and len(volumeSpec) == 2:
            data = volumeSpec[0]
            affine = volumeSpec[1]
        elif isinstance(volumeSpec, np.ndarray):
            data = volumeSpec
            affine = np.eye(4)
        elif isinstance(volumeSpec, str):
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
        if nanValue is not None:
            data[np.isnan(data)] = nanValue
        volume = Volume(data, affine)
        # add it to the volume list and obtain index
        # or obtain index of existing identical element
        if volume not in self.volumes:
            volumeID = len(self.volumes)
            self.volumes.append(volume)
        else:
            volumeID = self.volumes.index(volume)
        # indicate scene change
        self.sceneChanged = True
        # adjust camera position
        self.adjustCamera()
        # return the index as the volume ID
        return volumeID

    def adjustCamera(self):
        """
        change the camera position (not the direction)
        such that all volumes are in view,
        and the camera speed accordingly
        """
        # compute world positions of volume corners
        xyz = np.zeros((3, 0))
        for volume in self.volumes:
            shape = np.array(volume.data.shape)
            affine = volume.affine
            i, j, k = np.meshgrid([-0.5, shape[0] - 0.5],
                                  [-0.5, shape[1] - 0.5],
                                  [-0.5, shape[2] - 0.5], indexing='ij')
            ijk = np.vstack((i.flatten(), j.flatten(), k.flatten()))
            xyz = np.hstack((xyz, affine[:3, :3] @ ijk + affine[:3, 3, None]))
        # compute centroid
        c = np.mean(xyz, axis=1)
        # compute maximum distance from centroid
        d = max(np.sqrt(np.sum((xyz.T - c) ** 2, axis=1)))
        # set camera position & speed
        self.camPos = c - 2. * d * self._camDirections()[2]
        self.camSpeed = d / 3.

    def addSurface(self, volumeID, threshold, colorSpec):
        """
        add surface to the list of surfaces

        :param int volumeID:
            numeric ID of volume to use as basis, returned by addVolume
        :param float threshold:
            threshold applied to volume data
        :param colorSpec:
            color of the surface specified using matplotlib.colors syntax
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
        # indicate scene change
        self.sceneChanged = True

    # frame state --------------------------------------------------------------
    #
    # This includes everything whose change makes it necessary to render a new
    # frame, as well as timing.
    #
    # startTime:    time at which the scene was initialized
    # time:         time at current / last frame
    # frame:        number of current frame
    # camPos:       position of the camera in world space
    # camTheta:     azimuth of camera view (0 = positive x)
    # camPhi:       elevation of camera view (0 = horizontal)
    # camFovF:      field-of-view factor of camera view

    def _initializeFrameState(self):
        self.startTime = time.time()
        self.time = self.startTime
        self.frame = 0
        self.camTheta = np.radians(135.)        # similar to isometric view
        self.camPhi = -np.arctan(1 / np.sqrt(2))
        self.camPos = np.array([0, 0, 0])
        self.camFovF = np.tan(np.radians(30.))
        self.camSpeed = np.linalg.norm(self.camPos) / 5.

    def _camDirections(self):
        """
        directions of camera coordinate system

        :return: horizontal, vertical, center
        """
        # ray from the camera through the center of the frame
        center = np.array([np.cos(self.camTheta) * np.cos(self.camPhi),
                           np.sin(self.camTheta) * np.cos(self.camPhi),
                           np.sin(self.camPhi)])
        # horizontal direction in the frame:
        #  vector in the xy-plane that is orthogonal to the center ray
        horizontal = np.array([np.sin(self.camTheta),
                               -np.cos(self.camTheta),
                               0])
        # vertical direction in the frame: orthogonal to center & horizontal
        vertical = np.cross(horizontal, center)

        return horizontal, vertical, center

    @staticmethod
    def _deadzone(x, dead):
        """
        utility function for game controller input
        to neutralize small deviations from zero

        :param x:       raw gamecontroller input
        :param dead:    value up to which value should be set to 0
        :return:        "deadzoned" gamecontroller input
        """
        return np.copysign(max(0, np.abs(x) - dead) / (1 - dead), x)

    def _updateFrameState(self):
        """
        update frame state
        """

        # update time
        t = time.time() - self.startTime
        td = t - self.time
        self.time = t
        self.frame += 1

        # get controller input
        lt = sdl2.SDL_GameControllerGetAxis(
            self.gc, sdl2.SDL_CONTROLLER_AXIS_TRIGGERLEFT) / 32767
        rt = sdl2.SDL_GameControllerGetAxis(
            self.gc, sdl2.SDL_CONTROLLER_AXIS_TRIGGERRIGHT) / 32767
        ls = (np.array([
            sdl2.SDL_GameControllerGetAxis(
                self.gc, sdl2.SDL_CONTROLLER_AXIS_LEFTX),
            sdl2.SDL_GameControllerGetAxis(
                self.gc, sdl2.SDL_CONTROLLER_AXIS_LEFTY)]) + 0.5) / 32767.5
        rs = (np.array([
            sdl2.SDL_GameControllerGetAxis(
                self.gc, sdl2.SDL_CONTROLLER_AXIS_RIGHTX),
            sdl2.SDL_GameControllerGetAxis(
                self.gc, sdl2.SDL_CONTROLLER_AXIS_RIGHTY)]) + 0.5) / 32767.5
        du = sdl2.SDL_GameControllerGetButton(
            self.gc, sdl2.SDL_CONTROLLER_BUTTON_DPAD_UP)
        dd = sdl2.SDL_GameControllerGetButton(
            self.gc, sdl2.SDL_CONTROLLER_BUTTON_DPAD_DOWN)

        # update camera
        # camera direction
        if self.frame > 1:
            self.camTheta += -VoxelViewer._deadzone(rs[0], 0.25) * 2. * td
            self.camPhi += -VoxelViewer._deadzone(rs[1], 0.25) * 2. * td
        self.camPhi = np.clip(self.camPhi, -np.pi / 2, np.pi / 2)
        # directions of camera coordinate system
        horizontal, vertical, center = self._camDirections()
        # camera position
        if self.frame > 1:
            self.camPos += (VoxelViewer._deadzone(ls[0], 0.25) * horizontal
                            - VoxelViewer._deadzone(ls[1], 0.25) * center
                            + lt * vertical
                            - rt * vertical) * self.camSpeed * td
        # camera speed
        self.camSpeed *= 1.1 ** (du - dd)

    # --------------------------------------------------------------------------

    def __init__(self):
        # initialize scene state
        self._initializeSceneState()
        # initialize frame state
        self._initializeFrameState()

        # start event loop as thread
        self.thread = Thread(target=self._loop)
        self.thread.start()

    def _loop(self):
        # create environment
        self._createEnvironment()
        # create initial shader
        self._createShader()

        event = sdl2.SDL_Event()
        running = True
        while running:
            # has scene changed?
            if self.sceneChanged:
                # recreate shader
                self._destroyShader()
                self._createShader()

            # process keyboard and window events
            while sdl2.SDL_PollEvent(ctypes.byref(event)) != 0:
                if event.type == sdl2.SDL_QUIT:
                    # window close
                    running = False
                elif event.type == sdl2.SDL_KEYDOWN:
                    key = event.key
                    if key.repeat == 0:
                        if key.keysym.sym == sdl2.SDLK_ESCAPE:
                            # keyboard ESC
                            running = False
                        elif key.keysym.sym == sdl2.SDLK_f:
                            # keyboard f/F
                            self._toggleFullscreen()
                elif event.type == sdl2.SDL_CONTROLLERBUTTONDOWN:
                    button = event.cbutton.button
                    if button == sdl2.SDL_CONTROLLER_BUTTON_BACK:
                        # controller back
                        self.adjustCamera()
                    elif button == sdl2.SDL_CONTROLLER_BUTTON_START:
                        # controller start
                        self._toggleFullscreen()
            # update frame state
            self._updateFrameState()
            # render frame
            self._renderFrame()

            # give main thread (and other threads / processes) a chance to run
            #   1e-6 = 1 µs results in about 65 µs, that seems to be enough to
            # make the object responsive to method calls from the main thread.
            time.sleep(1e-6)

        # destroy shader
        self._destroyShader()
        # destroy environment
        self._destroyEnvironment()
