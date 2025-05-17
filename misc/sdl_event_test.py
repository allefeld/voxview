import sdl2
import ctypes

eventList = [           # see https://wiki.libsdl.org/SDL_EventType
    # Application events
    'SDL_QUIT',
    # Android, iOS and WinRT events
    'SDL_APP_TERMINATING', 'SDL_APP_LOWMEMORY', 'SDL_APP_WILLENTERBACKGROUND',
    'SDL_APP_DIDENTERBACKGROUND', 'SDL_APP_WILLENTERFOREGROUND',
    'SDL_APP_DIDENTERFOREGROUND',
    # Window events
    'SDL_WINDOWEVENT', 'SDL_SYSWMEVENT',
    # Keyboard events
    'SDL_KEYDOWN', 'SDL_KEYUP', 'SDL_TEXTEDITING', 'SDL_TEXTINPUT',
    'SDL_KEYMAPCHANGED',
    # Mouse events
    'SDL_MOUSEMOTION', 'SDL_MOUSEBUTTONDOWN', 'SDL_MOUSEBUTTONUP',
    'SDL_MOUSEWHEEL',
    # Joystick events
    'SDL_JOYAXISMOTION', 'SDL_JOYBALLMOTION', 'SDL_JOYHATMOTION',
    'SDL_JOYBUTTONDOWN', 'SDL_JOYBUTTONUP', 'SDL_JOYDEVICEADDED',
    'SDL_JOYDEVICEREMOVED',
    # Controller events
    'SDL_CONTROLLERAXISMOTION', 'SDL_CONTROLLERBUTTONDOWN',
    'SDL_CONTROLLERBUTTONUP', 'SDL_CONTROLLERDEVICEADDED',
    'SDL_CONTROLLERDEVICEREMOVED', 'SDL_CONTROLLERDEVICEREMAPPED',
    # Touch events
    'SDL_FINGERDOWN', 'SDL_FINGERUP', 'SDL_FINGERMOTION',
    # Gesture events
    'SDL_DOLLARGESTURE', 'SDL_DOLLARRECORD', 'SDL_MULTIGESTURE',
    # Clipboard events
    'SDL_CLIPBOARDUPDATE',
    # Drag and drop events
    'SDL_DROPFILE', 'SDL_DROPTEXT', 'SDL_DROPBEGIN', 'SDL_DROPCOMPLETE',
    # Audio hotplug events
    'SDL_AUDIODEVICEADDED', 'SDL_AUDIODEVICEREMOVED',
    # Render events
    'SDL_RENDER_TARGETS_RESET', 'SDL_RENDER_DEVICE_RESET']
eventDict = {}
for eventName in eventList:
    eventDict[eval('sdl2.' + eventName)] = eventName

sdl2.SDL_Init(sdl2.SDL_INIT_VIDEO | sdl2.SDL_INIT_GAMECONTROLLER)
window = sdl2.SDL_CreateWindow(
            b'',
            sdl2.SDL_WINDOWPOS_UNDEFINED, sdl2.SDL_WINDOWPOS_UNDEFINED,
            800, 600,
            sdl2.SDL_WINDOW_OPENGL | sdl2.SDL_WINDOW_RESIZABLE)
for i in range(sdl2.SDL_NumJoysticks()):
    if sdl2.SDL_IsGameController(i) == sdl2.SDL_TRUE:
        sdl2.SDL_GameControllerOpen(i)
        break

running = True
event = sdl2.SDL_Event()
while running:
    while sdl2.SDL_PollEvent(ctypes.byref(event)) != 0:
        eventName = eventDict[event.type]
        if event.type == sdl2.SDL_QUIT:
            # application event
            print(eventName)
            running = False
        elif (event.type == sdl2.SDL_KEYDOWN or
              event.type == sdl2.SDL_KEYUP):
            # keyboard event
            keyboardEvent = event.key
            keyName = sdl2.SDL_GetKeyName(keyboardEvent.keysym.sym)
            keyRepeat = keyboardEvent.repeat
            print(f"{eventName}:  \t{keyName} {keyRepeat}")
        elif (event.type == sdl2.SDL_CONTROLLERBUTTONDOWN or
              event.type == sdl2.SDL_CONTROLLERBUTTONUP):
            # controller button event
            buttonEvent = event.cbutton
            button = buttonEvent.button
            buttonName = sdl2.SDL_GameControllerGetStringForButton(button)
            print(f"{eventName}:    \t{buttonName}")
        else:
            # some other event
            print(eventName)

sdl2.SDL_Quit()
