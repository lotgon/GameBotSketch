import random
import time
from pywinauto import application
from pywinauto.findwindows import WindowAmbiguousError, WindowNotFoundError

APPS_POOL = ['Chrome', 'Notepad', 'Calculator']


# Init App object
app = application.Application()

app.connect(title_re=".*%s.*" % "New World")

# Access app's window object
app_dialog = app.top_window_()

app_dialog.move_window(0, 0)
app_dialog.minimize()
app_dialog.restore()
app_dialog.wait('visible')
time.sleep(1)
          
for i in range(10):
    img = app_dialog.capture_as_image()
    img.save("test%s.png"%i)
    time.sleep(1)
#app_dialog.SetFocus()
