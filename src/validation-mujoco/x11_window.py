"""Resize an X11 top-level window by title.

mujoco.viewer.launch_passive has no window size option (Simulate opens at 2/3 of
the monitor), so the window is resized after it appears, through libX11 directly.
"""
import ctypes
import ctypes.util
import time

_Window = ctypes.c_ulong


def _load_x11():
    path = ctypes.util.find_library("X11")
    if path is None:
        return None
    x11 = ctypes.cdll.LoadLibrary(path)
    x11.XOpenDisplay.argtypes = [ctypes.c_char_p]
    x11.XOpenDisplay.restype = ctypes.c_void_p
    x11.XDefaultRootWindow.argtypes = [ctypes.c_void_p]
    x11.XDefaultRootWindow.restype = _Window
    x11.XQueryTree.argtypes = [
        ctypes.c_void_p, _Window, ctypes.POINTER(_Window), ctypes.POINTER(_Window),
        ctypes.POINTER(ctypes.POINTER(_Window)), ctypes.POINTER(ctypes.c_uint)]
    x11.XFetchName.argtypes = [ctypes.c_void_p, _Window, ctypes.POINTER(ctypes.c_char_p)]
    x11.XFree.argtypes = [ctypes.c_void_p]
    x11.XResizeWindow.argtypes = [ctypes.c_void_p, _Window, ctypes.c_uint, ctypes.c_uint]
    x11.XFlush.argtypes = [ctypes.c_void_p]
    x11.XCloseDisplay.argtypes = [ctypes.c_void_p]
    return x11


def _title(x11, display, window):
    name = ctypes.c_char_p()
    if not x11.XFetchName(display, window, ctypes.byref(name)):
        return None
    if not name:
        return None
    title = name.value.decode("utf-8", errors="ignore")
    x11.XFree(ctypes.cast(name, ctypes.c_void_p))
    return title


def _children(x11, display, window):
    root = _Window()
    parent = _Window()
    children = ctypes.POINTER(_Window)()
    count = ctypes.c_uint()
    if not x11.XQueryTree(display, window, ctypes.byref(root), ctypes.byref(parent),
                          ctypes.byref(children), ctypes.byref(count)):
        return []
    result = [children[i] for i in range(count.value)]
    if children:
        x11.XFree(ctypes.cast(children, ctypes.c_void_p))
    return result


def _walk(x11, display, window):
    """Yield (window, title) for every window under `window`, depth first."""
    yield window, _title(x11, display, window)
    for child in _children(x11, display, window):
        yield from _walk(x11, display, child)


def window_titles():
    """All window titles on the display (for debugging)."""
    x11 = _load_x11()
    if x11 is None:
        return []
    display = x11.XOpenDisplay(None)
    if not display:
        return []
    try:
        root = x11.XDefaultRootWindow(display)
        return [title for _, title in _walk(x11, display, root) if title]
    finally:
        x11.XCloseDisplay(display)


def resize_window(title_prefix, width, height, timeout_s=3.0):
    """Resize the first window whose title starts with `title_prefix`. Returns True on success."""
    x11 = _load_x11()
    if x11 is None:
        return False
    display = x11.XOpenDisplay(None)
    if not display:
        return False

    try:
        root = x11.XDefaultRootWindow(display)
        deadline = time.monotonic() + timeout_s
        while True:
            for window, title in _walk(x11, display, root):
                if title is not None and title.startswith(title_prefix):
                    x11.XResizeWindow(display, window, width, height)
                    x11.XFlush(display)
                    return True
            if time.monotonic() > deadline:
                return False
            time.sleep(0.05)
    finally:
        x11.XCloseDisplay(display)
