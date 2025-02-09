import numpy as np
import cv2
import time
import pyautogui
from time import sleep
import Keys

def color_lines(original_image,lower,upper,threshold1,threshold2):
    image = original_image
    processed_img = original_image
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    maskr = cv2.inRange(hsv, lower, upper)
    processed_img = cv2.bitwise_and(processed_img,processed_img, mask =maskr)
    processed_img =  cv2.Canny(processed_img, threshold1, threshold2)
    return processed_img
    

def colorfind(image_BGR, HSVCOLOR_lower, HSVCOLOR_upper):
    hsv = cv2.cvtColor(image_BGR, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(hsv, HSVCOLOR_lower, HSVCOLOR_upper)
    processed_color = cv2.bitwise_and(image,image, mask=mask)
    return processed_color


def SW_SH_cals(my_x_pos,my_y_pos):
    SW, SH = pyautogui.size()
    ratiox=my_x_pos/2560    
    ratioy=my_y_pos/1600
    wherex=round(ratiox*SW)
    wherey=round(ratioy*SH)
    return wherex, wherey


def click(x,y,delay=0.1):
    xr,yr = SW_SH_cals(x,y)
    pyautogui.moveTo(xr,yr)
    pyautogui.mouseDown()
    pyautogui.mouseUp()
    time.sleep(delay)


def press(Key):
    Keys.PressKey(Key)
    Keys.ReleaseKey(Key)
    #time.sleep(0.1)


def open_chest_page():
    press(Keys.C)
    # can add to x depeding on page but I am just going to open first page most of the time    
    #pyautogui.moveTo(1492, 547)
    click(1492, 547)
    click(x=487, y=347)


def inventory_indexer(ind):
    ind_dict = {
        **{str(i): [[1753, 313], [1780 + (i % 4) * 200, 520 + (i // 4) * 200]] for i in range(16)},
        **{str(i): [[1937, 328], [1780 + (i % 4) * 200, 520 + ((i-16) // 4) * 200]] for i in range(16, 32)},
        **{str(i): [[2080, 329], [1780 + (i % 4) * 200, 520 + ((i-32) // 4) * 200]] for i in range(32, 48)},
        **{str(i): [[2283, 324], [1780 + (i % 4) * 200, 520 + ((i-48) // 4) * 200]] for i in range(48, 64)},
        **{str(i): [[2437, 328], [1780 + (i % 4) * 200, 520 + ((i-64) // 4) * 200]] for i in range(64, 80)}
    }
   
    click(ind_dict[ind][0][0],ind_dict[ind][0][1],0)
    pyautogui.moveTo(ind_dict[ind][1])

def hold_down():
    pyautogui.mouseDown()
    time.sleep(0.5)
    pyautogui.mouseUp()

if __name__ == "__main__":
    time.sleep(3)
    inventory_indexer(67)
