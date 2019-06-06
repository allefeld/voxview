#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# voxel viewer aka "brain game"
# CA 2019-6–1


import OpenGL.GL as gl
import sdl2
import numpy as np
import nibabel as nib
import time
import struct
import ctypes


# The shader must define
#   void mainImage(out vec4 color, in vec2 fragCoord)
# Without antialiasing, fragCoord.x / y
# goes from 0.5 to (width - 0.5) / (height - 0.5).
# For defined uniforms, see below `shaderHeader`.
#
# The shader can request anti-aliasing by `#define AA`,
# gamma correction by `#define GAMMA`,
# and error colors for debugging by `#define DEBUG`.

# fragment shader boilerplate
shaderHeader = """#version 430 core
uniform vec2  resolution;      // viewport resolution (in pixels)
uniform float time;            // shader playback time (in seconds)
uniform vec3  camPos;          // camera position
uniform mat3  camMatrix;       // camera matrix
// debugging
vec4 error = vec4(0., 0., 0., 0.);
#line 1 1
"""
shaderFooter = """
#line 1 2
out vec4 fragColor;
void main() {
    // anti-aliasing
    #ifdef AA
        // 4x super-sampling full-scene anti-aliasing
        fragColor = vec4(0., 0., 0., 0.);
        vec4 col;
        mainImage(gl_FragCoord.xy + vec2(0.25, 0.25),
                  col);
        fragColor += col;
        mainImage(gl_FragCoord.xy + vec2(-0.25, 0.25),
                  col);
        fragColor += col;
        mainImage(gl_FragCoord.xy + vec2(0.25, -0.25),
                  col);
        fragColor += col;
        mainImage(gl_FragCoord.xy + vec2(-0.25, -0.25),
                  col);
        fragColor += col;
        fragColor = fragColor / 4.0;
    #else
        mainImage(gl_FragCoord.xy,
                  fragColor);
    #endif
    // error-checking
    // if there are out-of-range color components, blink red/green
    #ifdef DEBUG
    if (error != vec4(0., 0., 0., 0.)) {
        if (fract(time * 2.) < 0.5) {
            fragColor = error;
        } else {
            fragColor = error * error.a;
        }
        return;
    }
    if (clamp(fragColor, 0., 1.) != fragColor) {
        fragColor = vec4(1., 1., 1., 1.);
        return;
    }
    #endif
    // conversion from linear to sRGB ("gamma correction")
    // adapted from https://www.shadertoy.com/view/lscSzl
    #ifdef GAMMA
        vec3 linearRGB = fragColor.rgb;
        vec3 a = 12.92 * linearRGB;
        vec3 b = 1.055 * pow(linearRGB, vec3(1.0 / 2.4)) - 0.055;
        vec3 c = step(vec3(0.0031308), linearRGB);
        fragColor = vec4(mix(a, b, c), fragColor.a);
    #endif
}
"""

# initialize SDL
sdl2.SDL_Init(sdl2.SDL_INIT_VIDEO | sdl2.SDL_INIT_GAMECONTROLLER)

# look for game controller
gc = None
for i in range(sdl2.SDL_NumJoysticks()):
    if sdl2.SDL_IsGameController(i) == sdl2.SDL_TRUE:
        gc = sdl2.SDL_GameControllerOpen(i)
        break
if gc is None:
    raise RuntimeError("No game controller found!")

# create window
window = sdl2.SDL_CreateWindow(
    "shader".encode('UTF-8'),
    sdl2.SDL_WINDOWPOS_UNDEFINED, sdl2.SDL_WINDOWPOS_UNDEFINED,
    800, 600,
    sdl2.SDL_WINDOW_OPENGL | sdl2.SDL_WINDOW_RESIZABLE)
# hide mouse
sdl2.SDL_ShowCursor(sdl2.SDL_DISABLE)
# # maximize window
# sdl2.SDL_MaximizeWindow(window)

# create OpenGL context
context = sdl2.SDL_GL_CreateContext(window)


# set up OpenGL context --------------------------------------------------------
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
quad = np.array([-1, -1,
                 1, -1,
                 -1, 1,
                 1, 1], dtype=np.float32)

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
    raise RuntimeError("OpenGL vertex shader could not be compiled\n"
                       + gl.glGetShaderInfoLog(vertexShader).decode('ASCII'))

# create and compile fragment shader
with open("voxview.glsl", 'r') as file:
    fragmentShaderSource = file.read()
fragmentShaderSource = shaderHeader + fragmentShaderSource + shaderFooter
fragmentShader = gl.glCreateShader(gl.GL_FRAGMENT_SHADER)
gl.glShaderSource(fragmentShader, fragmentShaderSource)
gl.glCompileShader(fragmentShader)
if gl.glGetShaderiv(fragmentShader, gl.GL_COMPILE_STATUS) == gl.GL_TRUE:
    print("*** OpenGL fragment shader compiled.")
else:
    raise RuntimeError("OpenGL fragment shader could not be compiled\n"
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
    raise RuntimeError("OpenGL program could not be linked\n"
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
# Here it is defined as consisting of pairs of GL_FLOAT type items with no
# other items between them (stride parameter 0) starting at offset 0
# in the buffer. This function refers to the currently bound GL_ARRAY_BUFFER,
# which is our vbo with the corner coordinates of the quad.
gl.glVertexAttribPointer(posAttrib, 2, gl.GL_FLOAT, False, 0, gl.GLvoidp(0))


# ------------------------------------------------------------------------------


filenames = ["mni_icbm152_nlin_asym_09c/mni_icbm152_gm_tal_nlin_asym_09c.nii",
             "mni_icbm152_nlin_asym_09c/mni_icbm152_wm_tal_nlin_asym_09c.nii",
             "sLPcomb-radek-X-C-PxC.nii"]
thresholds = [0.5, 0.5, 0.007]

for volID in range(len(filenames)):
    # load data
    img = nib.load(filenames[volID])
    # volume data
    data = img.get_fdata(dtype=np.float16)
    data[np.isnan(data)] = 0        # TODO
    # affine transformation
    #   rotation & scaling matrix (voxel -> world)
    AM = img.affine[:3, :3]
    # translation (voxel -> world), world position of voxel [0, 0, 0]
    AO = img.affine[:3, 3]
    AO[0] += volID * 100 - 100
    # inverse rotation & scaling (world -> voxel)
    AiM = np.linalg.inv(AM)

    # pass volume data as texture -> uniform sampler3D
    texture = gl.glGenTextures(1)
    gl.glUniform1i(
        gl.glGetUniformLocation(program, "vol[%d].data" % volID),
        volID)
    gl.glActiveTexture(gl.GL_TEXTURE0 + volID)
    gl.glBindTexture(gl.GL_TEXTURE_3D, texture)
    gl.glTexImage3D(gl.GL_TEXTURE_3D, 0, gl.GL_R16F, *data.shape,
                    0, gl.GL_RED, gl.GL_FLOAT, data.flatten('F'))
    # should the following two be necessary when using texelFetch?!
    gl.glTexParameteri(gl.GL_TEXTURE_3D, gl.GL_TEXTURE_MAX_LEVEL, 0)
    gl.glClampColor(gl.GL_CLAMP_READ_COLOR, gl.GL_FALSE)
    gl.glEnable(gl.GL_TEXTURE_3D)

    # pass affine transformation
    gl.glUniformMatrix3fv(
        gl.glGetUniformLocation(program, "vol[%d].AM" % volID),
        1, gl.GL_FALSE, *AM.flatten('F'))
    gl.glUniform3f(
        gl.glGetUniformLocation(program, "vol[%d].AO" % volID),
        *AO)
    gl.glUniformMatrix3fv(
        gl.glGetUniformLocation(program, "vol[%d].AiM" % volID),
        1, gl.GL_FALSE, struct.pack('f' * 9, *AiM.flatten('F')))
    # Why is struct.pack necessary for glUniformMatrix3fv,
    # but not for glUniform3f?

for surfID in range(len(filenames)):
    # define surface
    volID = surfID
    threshold = thresholds[volID]
    # http://devernay.free.fr/cours/opengl/materials.html
    if surfID == 0:         # gray
        color = [0.1, 0.1, 0.1]
    elif surfID == 1:       # white
        color = [1., 1., 1.]
    elif surfID == 2:       # red
        color = [1., 0., 0.]
    ka = np.array(color) * 0.3
    kd = np.array(color) * 0.3
    ks = np.array([1., 1., 1.]) * 0.1
    alpha = 10.

    # pass volume ID
    gl.glUniform1i(
        gl.glGetUniformLocation(program, "surf[%d].volID" % surfID),
        volID)
    # pass threshold
    gl.glUniform1f(
        gl.glGetUniformLocation(program, "surf[%d].threshold" % surfID),
        threshold)
    # pass Phong coefficients
    gl.glUniform3f(
        gl.glGetUniformLocation(program, "surf[%d].ka" % surfID),
        *ka)
    gl.glUniform3f(
        gl.glGetUniformLocation(program, "surf[%d].kd" % surfID),
        *kd)
    gl.glUniform3f(
        gl.glGetUniformLocation(program, "surf[%d].ks" % surfID),
        *ks)
    gl.glUniform1f(
        gl.glGetUniformLocation(program, "surf[%d].alpha" % surfID),
        alpha)


# ------------------------------------------------------------------------------


# *** initialize timing & state ***
startTime = time.time()
lastTime = float('nan')
frame = 0
camPos = np.array([0., 200., 0.])
camTheta = np.pi/2
camPhi = 0


def deadzone(x, dead):
    return np.copysign(max(0, np.abs(x) - dead) / (1 - dead), x)


def display():
    """redraw window"""

    # *** update window
    w = ctypes.c_int()
    h = ctypes.c_int()
    sdl2.SDL_GetWindowSize(window, w, h)
    gl.glViewport(0, 0, w, h)

    # *** update time & input
    # timing
    t = time.time() - startTime
    global lastTime
    td = t - lastTime
    lastTime = t
    global frame
    frame += 1
    # controller
    # lt = sdl2.SDL_GameControllerGetAxis(
    #     gc, sdl2.SDL_CONTROLLER_AXIS_TRIGGERLEFT) / 32767
    rt = sdl2.SDL_GameControllerGetAxis(
        gc, sdl2.SDL_CONTROLLER_AXIS_TRIGGERRIGHT) / 32767
    ls = (np.array([
        sdl2.SDL_GameControllerGetAxis(
            gc, sdl2.SDL_CONTROLLER_AXIS_LEFTX),
        sdl2.SDL_GameControllerGetAxis(
            gc, sdl2.SDL_CONTROLLER_AXIS_LEFTY)]) + 0.5) / 32767.5
    rs = (np.array([
        sdl2.SDL_GameControllerGetAxis(
            gc, sdl2.SDL_CONTROLLER_AXIS_RIGHTX),
        sdl2.SDL_GameControllerGetAxis(
            gc, sdl2.SDL_CONTROLLER_AXIS_RIGHTY)]) + 0.5) / 32767.5
    du = sdl2.SDL_GameControllerGetButton(
        gc, sdl2.SDL_CONTROLLER_BUTTON_DPAD_UP)
    dd = sdl2.SDL_GameControllerGetButton(
        gc, sdl2.SDL_CONTROLLER_BUTTON_DPAD_DOWN)

    # *** update state
    # camera direction
    global camTheta, camPhi
    if frame > 1:
        camTheta += deadzone(rs[0], 0.25) * 0.05
        camPhi += -deadzone(rs[1], 0.25) * 0.05
    camPhi = np.clip(camPhi, -np.pi/2, np.pi/2)
    #   ray from the camera through the center of the frame
    center = np.array([np.cos(camTheta) * np.cos(camPhi),
                       -np.sin(camTheta) * np.cos(camPhi),
                       np.sin(camPhi)])
    #  horizontal direction in the frame:
    #  vector in the xy-plane that is orthogonal to the center ray
    horizontal = np.array([-np.sin(camTheta),
                           -np.cos(camTheta),
                           0])
    #  vertical direction in the frame: orthogonal to both center & horizontal
    vertical = np.cross(horizontal, center)
    # camera position
    global camPos
    if frame > 1:
        camPos += (deadzone(ls[0], 0.25) * horizontal
                   - deadzone(ls[1], 0.25) * center
                   + du * vertical
                   - dd * vertical) * td * 20
    # camera field of view -> zoom
    fovf = np.tan(np.radians(30 * (1 - rt)))
    horizontal *= fovf
    vertical *= fovf

    # *** update uniforms
    # update generic uniforms
    gl.glUniform2f(
        gl.glGetUniformLocation(program, "resolution"),
        float(w.value), float(h.value))
    gl.glUniform1f(
        gl.glGetUniformLocation(program, "time"),
        t)
    # update camera position uniform
    gl.glUniform3f(
        gl.glGetUniformLocation(program, "camPos"),
        *camPos)
    # update camera matrix uniform
    cm = np.column_stack((horizontal, vertical, center))
    gl.glUniformMatrix3fv(
        gl.glGetUniformLocation(program, "camMatrix"), 1, gl.GL_FALSE,
        struct.pack('f'*9, *cm.flatten('F')))

    # use vertex buffer data to draw
    #   The first four values of the vertex sequence are interpreted as
    # specifying a triangle strip (see above).
    gl.glClearColor(1.0, 0.5, 0.0, 1.0)
    gl.glClear(gl.GL_COLOR_BUFFER_BIT)
    gl.glDrawArrays(gl.GL_TRIANGLE_STRIP, 0, 4)

    # time display updates
    title = "shader:  %d - %.2f - %.2f - %.2f" % (frame, t, frame / t, 1 / td)
    sdl2.SDL_SetWindowTitle(window, title.encode('UTF-8'))

    # swap buffers (glFlush + glXSwapBuffers?)
    # possibly use glFinish() to make timing tighter – doesn't seem to hurt much
    gl.glFinish()
    sdl2.SDL_GL_SwapWindow(window)


# ------------------------------------------------------------------------------


def toggleFullscreen():
    """toggle between windowed and fullscreen mode"""
    if sdl2.SDL_GetWindowFlags(window) & sdl2.SDL_WINDOW_FULLSCREEN_DESKTOP:
        sdl2.SDL_SetWindowFullscreen(window, 0)
    else:
        sdl2.SDL_SetWindowFullscreen(
            window, sdl2.SDL_WINDOW_FULLSCREEN_DESKTOP)


# event loop
# TODO: we want rendering to be paused when the window is invisible
# also, the current event loop seems to knock out part of KDE
event = sdl2.SDL_Event()
running = True
while running:
    # process keyboard and window events
    while sdl2.SDL_PollEvent(ctypes.byref(event)) != 0:
        if event.type == sdl2.SDL_QUIT:
            running = False
        elif event.type == sdl2.SDL_KEYDOWN:
            if event.key.keysym.sym == sdl2.SDLK_ESCAPE:
                running = False
            elif event.key.keysym.sym == sdl2.SDLK_f:
                toggleFullscreen()
    # render frame
    display()
    # give the CPU some rest?
    # sdl2.SDL_Delay(10)


# cleanup
gl.glDisableVertexAttribArray(posAttrib)
gl.glDeleteProgram(program)
gl.glDeleteShader(fragmentShader)
gl.glDeleteShader(vertexShader)
gl.glDeleteBuffers(1, [VBO])
gl.glDeleteVertexArrays(1, [VAO])
sdl2.SDL_GL_DeleteContext(context)
sdl2.SDL_DestroyWindow(window)
sdl2.SDL_Quit()
