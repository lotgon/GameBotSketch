from pywinauto import application
from pywinauto.findwindows import WindowAmbiguousError, WindowNotFoundError
import time

class Capturer(object):
    """description of class"""
    def __init__(self, pattern_name):
        # Init App object
        self.app = application.Application()
        self.app.connect(title_re=pattern_name)
        self.app_dialog = self.app.top_window()
        self.wrapper_object = self.app_dialog.wrapper_object()
    
    def ShowWindow(self):
        self.app_dialog.minimize()
        self.app_dialog.restore()
        self.app_dialog.wait('visible')
        time.sleep(1)
    def Capture(self):
        return( self.app_dialog.capture_as_image())
    def WindowsPosition(self):
        r = self.wrapper_object.rectangle()
        return (r.left, r.top, r.right, r.bottom)

