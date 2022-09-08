import pygetwindow
import time
import os
import pyautogui
import PIL
from pygetwindow import PyGetWindowException

pyautogui.position() 
pyautogui.screenshot("Test2.png")
pyautogui.moveTo(100, 100, 2, pyautogui.easeInQuad)


# get screensize
x,y = pyautogui.size()
print(f"width={x}\theight={y}")

x2,y2 = pyautogui.size()
x2,y2=int(str(x2)),int(str(y2))
print(x2//2)
print(y2//2)

z1 = pygetwindow.getAllTitles()

my = pygetwindow.getWindowsWithTitle(z1[5])[0]
# quarter of screen screensize
#my.resizeTo(x3,y3)
# top-left
#my.moveTo(0, 0)
my.moveTo(0, 0)
time.sleep(3)
my.activate()

pyautogui.screenshot("Test2.png")
