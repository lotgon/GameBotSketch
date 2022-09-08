from pywinauto import keyboard
import time

keyboard.send_keys("e")


from mouse import WindMouse 
def mouseFigure():
    while True:
        print(WindMouse.get_mouse_position())
        WindMouse.wind_mouse_straight(-1, 0)
        print(WindMouse.get_mouse_position())
        time.sleep(5)
        #WindMouse.wind_mouse(0, 100)
        #WindMouse.wind_mouse(100, 0)
        #WindMouse.wind_mouse(0, -100)
mouseFigure()
