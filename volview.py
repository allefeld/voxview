#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# volume viewer aka "brain game"
# CA 2019-5-31


import sdl2
from OpenGL.GL import *
import numpy as np
import time
import struct
import nibabel as nib


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
        mainImage(col, gl_FragCoord.xy + vec2(0.25, 0.25));
        fragColor += col;
        mainImage(col, gl_FragCoord.xy + vec2(-0.25, 0.25));
        fragColor += col;
        mainImage(col, gl_FragCoord.xy + vec2(0.25, -0.25));
        fragColor += col;
        mainImage(col, gl_FragCoord.xy + vec2(-0.25, -0.25));
        fragColor += col;
        fragColor = fragColor / 4.0;
    #else
        mainImage(fragColor, gl_FragCoord.xy);
    #endif
    // error-checking
    // if there are out-of-range color components, blink red/green
    #ifdef DEBUG
    if (clamp(fragColor, 0., 1.) != fragColor) {
        error = vec4(1., 0., 0., 1.);
    }
    if (error != vec4(0., 0., 0., 0.)) {
        if (fract(time * 2.) < 0.5) {
            fragColor = error;
        } else {
            fragColor = error * error.a;
        }
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
VAO = glGenVertexArrays(1)
glBindVertexArray(VAO)

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
VBO = glGenBuffers(1)
# initialize the VBO to be used for vertex data
glBindBuffer(GL_ARRAY_BUFFER, VBO)
# create mutable storage for the VBO and copy data into it
glBufferData(GL_ARRAY_BUFFER, quad, GL_STATIC_DRAW)

# create and compile vertex shader
#   This vertex shader simply puts vertices at our 2D positions.
vertexShaderSource = """#version 100
    in vec2 position;
    void main() {
        gl_Position = vec4(position, 0., 1.);
    }"""
vertexShader = glCreateShader(GL_VERTEX_SHADER)
glShaderSource(vertexShader, vertexShaderSource)
glCompileShader(vertexShader)
if glGetShaderiv(vertexShader, GL_COMPILE_STATUS) == GL_TRUE:
    print("*** OpenGL vertex shader compiled.")
else:
    raise RuntimeError("OpenGL vertex shader could not be compiled\n"
                       + glGetShaderInfoLog(vertexShader).decode('ASCII'))

# create and compile fragment shader
with open("volview.glsl", 'r') as file:
    fragmentShaderSource = file.read()
fragmentShaderSource = shaderHeader + fragmentShaderSource + shaderFooter
fragmentShader = glCreateShader(GL_FRAGMENT_SHADER)
glShaderSource(fragmentShader, fragmentShaderSource)
glCompileShader(fragmentShader)
if glGetShaderiv(fragmentShader, GL_COMPILE_STATUS) == GL_TRUE:
    print("*** OpenGL fragment shader compiled.")
else:
    raise RuntimeError("OpenGL fragment shader could not be compiled\n"
                       + glGetShaderInfoLog(fragmentShader).decode('ASCII'))

# create program object and attach shaders
program = glCreateProgram()
glAttachShader(program, vertexShader)
glAttachShader(program, fragmentShader)
# make name of fragment shader color output explicit
glBindFragDataLocation(program, 0, b"fragColor")
# link the program
glLinkProgram(program)
if glGetProgramiv(program, GL_LINK_STATUS) == GL_TRUE:
    print("*** OpenGL program linked.")
else:
    raise RuntimeError("OpenGL program could not be linked\n"
                       + glGetProgramInfoLog(program).decode('ASCII'))
# validate the program
glValidateProgram(program)
# activate the program
glUseProgram(program)

# specify the layout of our vertex data
#   get a handle for the input variable position in our shader program
posAttrib = glGetAttribLocation(program, b"position")
#   activate this input
glEnableVertexAttribArray(posAttrib)
#   format of the vertex data
# Here it is defined as consisting of pairs of GL_FLOAT type items with no
# other items between them (stride parameter 0) starting at offset 0
# in the buffer. This function refers to the currently bound GL_ARRAY_BUFFER,
# which is our vbo with the corner coordinates of the quad.
glVertexAttribPointer(posAttrib, 2, GL_FLOAT, False, 0, ctypes.c_voidp(0))


# ------------------------------------------------------------------------------

# load data
img = nib.load("mni_icbm152_nlin_asym_09c/mni_icbm152_gm_tal_nlin_asym_09c.nii")
# volume data
vol = img.get_fdata(dtype=np.float16)
vol[0, 0, 0] = 1                # left–posterior–inferior-most marker voxel
# threshold
threshold = 0.15
# affine transformation
#   rotation & scaling matrix (voxel -> world)
AM = img.affine[:3, :3]
# translation (voxel -> world), position of voxel [0, 0, 0]
AO = img.affine[:3, 3]
#   inverse rotation & scaling (world -> voxel)
AiM = np.linalg.inv(AM)

# pass volume data as texture -> uniform sampler3D
texture = glGenTextures(1)
glUniform1i(glGetUniformLocation(program, "vol"), 0)
glActiveTexture(GL_TEXTURE0 + 0)
glBindTexture(GL_TEXTURE_3D, texture)
glTexImage3D(GL_TEXTURE_3D, 0, GL_R16F,
             vol.shape[0], vol.shape[1], vol.shape[2],
             0, GL_RED, GL_FLOAT, vol.flatten('F'))
# should the following two be necessary when using texelFetch?!
glTexParameteri(GL_TEXTURE_3D, GL_TEXTURE_MAX_LEVEL, 0)
glClampColor(GL_CLAMP_READ_COLOR, GL_FALSE)
glEnable(GL_TEXTURE_3D)

# pass threshold
glUniform1f(glGetUniformLocation(program, "threshold"), threshold)

# pass affine transformation
glUniformMatrix3fv(glGetUniformLocation(program, "AM"), 1, GL_FALSE,
                   struct.pack('f' * 9, *AM.flatten('F')))
glUniform3f(glGetUniformLocation(program, "AO"), *AO)
glUniformMatrix3fv(glGetUniformLocation(program, "AiM"), 1, GL_FALSE,
                   struct.pack('f' * 9, *AiM.flatten('F')))

# ------------------------------------------------------------------------------


# *** initialize timing & state ***
startTime = time.time()
lastTime = float('nan')
frame = 0
# camPos = np.array([-vol.shape[0], vol.shape[1] / 2, vol.shape[2] / 2])
camPos = np.array([0., 200., 0.])
camTheta = np.pi/2
camPhi = 0


def deadzone(x, dead):
    return np.copysign(max(0, np.abs(x) - dead) / (1 - dead), x)


def display():
    """redraw window callbacks"""

    # *** update window
    w = ctypes.c_int()
    h = ctypes.c_int()
    sdl2.SDL_GetWindowSize(window, w, h)
    glViewport(0, 0, w, h)

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
                   - dd * vertical) * td * 50
    # camera field of view -> zoom
    fovf = np.tan(np.radians(30 * (1 - rt)))
    horizontal *= fovf
    vertical *= fovf

    # *** update uniforms
    # update generic uniforms
    glUniform2f(glGetUniformLocation(program, "resolution"),
                float(w.value), float(h.value))
    glUniform1f(glGetUniformLocation(program, "time"), t)
    # update camera position uniform
    glUniform3f(glGetUniformLocation(program, "camPos"), *camPos)
    # update camera matrix uniform
    cm = np.column_stack((horizontal, vertical, center))
    glUniformMatrix3fv(glGetUniformLocation(program, "camMatrix"), 1, GL_FALSE,
                       struct.pack('f'*9, *cm.flatten('F')))

    # use vertex buffer data to draw
    #   The first four values of the vertex sequence are interpreted as
    # specifying a triangle strip (see above).
    glClearColor(1.0, 0.5, 0.0, 1.0)
    glClear(GL_COLOR_BUFFER_BIT)
    glDrawArrays(GL_TRIANGLE_STRIP, 0, 4)

    # time display updates
    title = "shader:  %d - %.2f - %.2f - %.2f" % (frame, t, frame / t, 1 / td)
    sdl2.SDL_SetWindowTitle(window, title.encode('UTF-8'))


# ------------------------------------------------------------------------------


# toggle between windowed and fullscreen mode
def toggleFullscreen():
    """toggle between windowed and fullscreen mode"""
    if sdl2.SDL_GetWindowFlags(window) & sdl2.SDL_WINDOW_FULLSCREEN_DESKTOP:
        sdl2.SDL_SetWindowFullscreen(window, 0)
    else:
        sdl2.SDL_SetWindowFullscreen(
            window, sdl2.SDL_WINDOW_FULLSCREEN_DESKTOP)


# create dictionary that maps event.window.event to names
windowEventNames = [
    'SDL_WINDOWEVENT_SHOWN', 'SDL_WINDOWEVENT_HIDDEN',
    'SDL_WINDOWEVENT_EXPOSED', 'SDL_WINDOWEVENT_MOVED',
    'SDL_WINDOWEVENT_RESIZED', 'SDL_WINDOWEVENT_SIZE_CHANGED',
    'SDL_WINDOWEVENT_MINIMIZED', 'SDL_WINDOWEVENT_MAXIMIZED',
    'SDL_WINDOWEVENT_RESTORED', 'SDL_WINDOWEVENT_ENTER',
    'SDL_WINDOWEVENT_LEAVE', 'SDL_WINDOWEVENT_FOCUS_GAINED',
    'SDL_WINDOWEVENT_FOCUS_LOST', 'SDL_WINDOWEVENT_CLOSE',
    'SDL_WINDOWEVENT_TAKE_FOCUS', 'SDL_WINDOWEVENT_HIT_TEST']
windowEventName = {}
for e in windowEventNames:
    windowEventName[eval('sdl2.' + e)] = e


# TODO: we want rendering to be paused when the window is invisible


# event loop
event = sdl2.SDL_Event()
running = True
while running:
    while sdl2.SDL_PollEvent(ctypes.byref(event)) != 0:
        if event.type == sdl2.SDL_QUIT:
            running = False
        elif event.type == sdl2.SDL_KEYDOWN:
            if event.key.keysym.sym == sdl2.SDLK_ESCAPE:
                running = False
            elif event.key.keysym.sym == sdl2.SDLK_f:
                toggleFullscreen()
        # elif event.type == sdl2.SDL_WINDOWEVENT:
        #     print(windowEventName[event.window.event],
        #           event.window.data1, event.window.data2)

    display()

    # swap buffers (glFlush + glXSwapBuffers?)
    # possibly use glFinish() to make timing tighter – doesn't seem to hurt much
    sdl2.SDL_GL_SwapWindow(window)

    sdl2.SDL_Delay(10)


# cleanup
glDisableVertexAttribArray(posAttrib)
glDeleteProgram(program)
glDeleteShader(fragmentShader)
glDeleteShader(vertexShader)
glDeleteBuffers(1, [VBO])
glDeleteVertexArrays(1, [VAO])
sdl2.SDL_GL_DeleteContext(context)
sdl2.SDL_DestroyWindow(window)
sdl2.SDL_Quit()


# eventNames = ['SDL_FIRSTEVENT', 'SDL_QUIT', 'SDL_APP_TERMINATING',
#               'SDL_APP_LOWMEMORY', 'SDL_APP_WILLENTERBACKGROUND',
#               'SDL_APP_DIDENTERBACKGROUND','SDL_APP_WILLENTERFOREGROUND',
#               'SDL_APP_DIDENTERFOREGROUND', 'SDL_WINDOWEVENT',
#               'SDL_SYSWMEVENT', 'SDL_KEYDOWN', 'SDL_KEYUP',
#               'SDL_TEXTEDITING', 'SDL_TEXTINPUT', 'SDL_KEYMAPCHANGED',
#               'SDL_MOUSEMOTION', 'SDL_MOUSEBUTTONDOWN', 'SDL_MOUSEBUTTONUP',
#               'SDL_MOUSEWHEEL', 'SDL_JOYAXISMOTION', 'SDL_JOYBALLMOTION',
#               'SDL_JOYHATMOTION', 'SDL_JOYBUTTONDOWN', 'SDL_JOYBUTTONUP',
#               'SDL_JOYDEVICEADDED', 'SDL_JOYDEVICEREMOVED',
#               'SDL_CONTROLLERAXISMOTION', 'SDL_CONTROLLERBUTTONDOWN',
#               'SDL_CONTROLLERBUTTONUP', 'SDL_CONTROLLERDEVICEADDED',
#               'SDL_CONTROLLERDEVICEREMOVED', 'SDL_CONTROLLERDEVICEREMAPPED',
#               'SDL_FINGERDOWN', 'SDL_FINGERUP', 'SDL_FINGERMOTION',
#               'SDL_DOLLARGESTURE', 'SDL_DOLLARRECORD', 'SDL_MULTIGESTURE',
#               'SDL_CLIPBOARDUPDATE', 'SDL_DROPFILE', 'SDL_DROPTEXT',
#               'SDL_DROPBEGIN', 'SDL_DROPCOMPLETE', 'SDL_AUDIODEVICEADDED',
#               'SDL_AUDIODEVICEREMOVED', 'SDL_RENDER_TARGETS_RESET',
#               'SDL_RENDER_DEVICE_RESET', 'SDL_USEREVENT', 'SDL_LASTEVENT']
