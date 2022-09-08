import numpy as np
from pywinauto import mouse
import win32api

class WindMouse(object):

    sqrt3 = np.sqrt(3)
    sqrt5 = np.sqrt(5)
    
    @staticmethod
    def get_mouse_position():
        return win32api.GetCursorPos()

    @staticmethod
    def wind_mouse(move_x, move_y, G_0=9, W_0=3, M_0=30, D_0=20, move_mouse=lambda x,y: mouse.move(coords=(x, y))):

        #WindMouse algorithm. Calls the move_mouse kwarg with each new step.
        #Released under the terms of the GPLv3 license.
        #G_0 - magnitude of the gravitational fornce
        #W_0 - magnitude of the wind force fluctuations
        #M_0 - maximum step size (velocity clip threshold)
        #D_0 - distance where wind behavior changes from random to damped

        start_x, start_y = win32api.GetCursorPos()
        dest_x, dest_y = start_x + move_x, start_y + move_y
        current_x,current_y = start_x,start_y
        v_x = v_y = W_x = W_y = 0

        counter = 0
        while True:
            dist=np.hypot(dest_x-start_x,dest_y-start_y)
            if dist <1:
                break
            W_mag = min(W_0, dist)
            if dist >= D_0:
                W_x = W_x/WindMouse.sqrt3 + (2*np.random.random()-1)*W_mag/WindMouse.sqrt5
                W_y = W_y/WindMouse.sqrt3 + (2*np.random.random()-1)*W_mag/WindMouse.sqrt5
            else:
                W_x /= WindMouse.sqrt3
                W_y /= WindMouse.sqrt3
                if M_0 < 3:
                    M_0 = np.random.random()*3 + 3
                else:
                    M_0 /= WindMouse.sqrt5
            v_x += W_x + G_0*(dest_x-start_x)/dist
            v_y += W_y + G_0*(dest_y-start_y)/dist
            v_mag = np.hypot(v_x, v_y)
            if v_mag > M_0:
                v_clip = M_0/2 + np.random.random()*M_0/2
                v_x = (v_x/v_mag) * v_clip
                v_y = (v_y/v_mag) * v_clip
            start_x += v_x
            start_y += v_y
            move_x = int(np.round(start_x))
            move_y = int(np.round(start_y))
            if current_x != move_x or current_y != move_y:
                #This should wait for the mouse polling interval
                current_x=move_x
                current_y=move_y
                counter+=1
                #if counter%2 == 0:
                move_mouse(current_x, current_y)
                print("Mouse position: %s, %s"%(current_x, current_y))
        return current_x,current_y
    @staticmethod
    def wind_mouse_straight(move_x, move_y, G_0=9, W_0=3, M_0=30, D_0=20, move_mouse=lambda x,y: mouse.move(coords=(x, y))):

        #WindMouse algorithm. Calls the move_mouse kwarg with each new step.
        #Released under the terms of the GPLv3 license.
        #G_0 - magnitude of the gravitational fornce
        #W_0 - magnitude of the wind force fluctuations
        #M_0 - maximum step size (velocity clip threshold)
        #D_0 - distance where wind behavior changes from random to damped

        start_x, start_y = win32api.GetCursorPos()
        dest_x, dest_y = start_x + move_x, start_y + move_y

        move_mouse(dest_x, dest_y)
        print("Mouse position: %s, %s"%(dest_x, dest_y))
        return dest_x,dest_y
