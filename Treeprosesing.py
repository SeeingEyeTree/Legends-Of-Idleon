##Start
import cv2
import numpy as np
import matplotlib.pyplot as plt
import time
from PIL import ImageGrab
import pyautogui
from Keybordinput import PressKey,ReleaseKey, W, A, S, D, SPACE, I, ONE

leaf=cv2.imread('E:\\IsThisAlowed\\images\\nobackgrounleaf.png')
leafgray=cv2.cvtColor(leaf, cv2.COLOR_BGR2GRAY)

w0, h0 = leafgray.shape[::-1]


def imageprocessing_for_redlines(original_image):
    image = original_image
    processed_img = original_image
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    lower_red=np.array([120,50,50])
    upper_red=np.array([255,250,250])
    maskr = cv2.inRange(hsv, lower_red, upper_red)
    processed_imgR = cv2.bitwise_and(processed_img,processed_img, mask =maskr)
    processed_imgR =  cv2.Canny(processed_imgR, threshold1 = 100, threshold2=200)
    #ret, processed_img = cv2.threshold(processed_img, 150, 255, cv2.THRESH_BINARY_INV)
    return processed_imgR

def imageprocessing_for_Black(original_image):
    image = original_image
    processed_img = original_image
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    lower_Black=np.array([80,80,80])
    upper_Black=np.array([127 ,167 ,193])
    maskr = cv2.inRange(hsv, lower_Black, upper_Black)
    processed_imgB = cv2.bitwise_and(processed_img,processed_img, mask =maskr)
    processed_imgB =  cv2.Canny(processed_imgB, threshold1 = 2, threshold2=2)
    #ret, processed_img = cv2.threshold(processed_img, 150, 255, cv2.THRESH_BINARY_INV)
    return processed_imgB

def rowHasColorVal(row):
    for col in row:
        #print('processing column',col)
        if (col>0):
            return 1
    return 0

#red_x_vals=np.insert[xvalues_for_red,xposr]


def colsWithRed(row):
    xpos=0
    redlist=[]
    #print('Getting cols with red from ',row)
    for col in row:
        #print('Processing x',xpos)       
        if (col>0):
           #print('column is red ',xpos)
           redlist.append(xpos)
        xpos=xpos+1
        #print('xpos:',xpos)
    return redlist


def screen_record(): 
    x=1

    n=0
    while(True):
    #while(n < 1):
        n=n+1
        printscreen =  np.array(ImageGrab.grab(bbox=(1200,180,1700,240)))
        printscreen1=np.array(ImageGrab.grab(bbox=(1200,180,1700,210)))
        printscreenRGB=cv2.cvtColor(printscreen,cv2.COLOR_BGR2RGB)
        printscreenGRAY = cv2.cvtColor(printscreen, cv2.COLOR_BGR2GRAY)
        redlines=imageprocessing_for_redlines(printscreen)
        leaf=imageprocessing_for_Black(printscreen1)
        res0 = cv2.matchTemplate(printscreenGRAY,leafgray,cv2.TM_CCOEFF_NORMED)
        threshold0 = 0.8
        loc0 = np.where( res0 >= threshold0)
        for pt0 in zip(*loc0[::-1]):
            cv2.rectangle(printscreenRGB, pt0, (pt0[0] + w0, pt0[1] + h0), (0,255,0), 2)
        cv2.imshow('redlins',redlines)
        cv2.imshow('black', leaf)
        xposl=0
        xposr=0
        xvalues_for_leaf=np.zeros(500)
        #np.delete(xvalues_for_leaf,axis=0)
        xvalues_for_red=np.zeros(500)
        #np.delete(xvalues_for_red,axis=0)
        i=0
        #redlines=np.array([[0,0,0,0,0],[0,0,255,0,0],[255,255,255,255,255],[255,0,0,0,0]])
       # print('Going through leaf array')
        #print('leaf size',leaf.size)
        #print('leaf shape',leaf.shape)
        for rows in leaf:
            #print('rows')
            #print(rows)
            #print('rows size',rows.size)
            #print('rows shape',rows.shape)
            
            leaft=rowHasColorVal(rows)
            if (leaft > 0) :
                leafList=colsWithRed(rows)
                leaf_x_vals=np.array(leafList)  
                #print(leaf_x_vals)      

        
        #print('Going through redlines array')
        for rowArray in redlines:
            #print('rowArray')
            #print(rowArray)
            redLine=rowHasColorVal(rowArray)
            if (redLine > 0) :
                #print('Found non zero in row')
                redList=colsWithRed(rowArray)
                red_x_vals=np.array(redList)
                #print('Row with red values are')
                #print(red_x_vals)

        if leaf_x_vals.all() != red_x_vals.all():
            PressKey(SPACE)
            ReleaseKey(SPACE)
        
            #if (rowArray==255).[xposr]: 
            #    red_x_vals=np.insert[xvalues_for_red,xposr]
            #    xpoxr=xposr+1
            #xposr=xposr+1
            
            #i=i+1
            #if ( i > 2) :
            #  break
        
        #for column in black:
        #    if (column>0).[xposl]:
        #        red_x_vals=np.insert[xvalues_for_red,xposl]
        #        xpoxl=xposl+1
        #    xposl=xposl+1
            
        #file = open("E:\\IsThisAlowed\\Text\\redpixs.txt","w")
        
        #for column in xvalues_for_red:
        #    np.savetxt(file, row)
        #indexA=np.array[]
        #lastVal=0
        #currI=0
        #columnI=0
        #for x in redlines
         #   if x != lastVal:
          #      indexA[0,currI]=columnI
           #     currI=currI + 1
            #lastVal=x
            #columnI=columnI+1
        #    if x>0:
        #        if i1 == 0:
        #            i1=x  
        #        i2=x
                
        
        #cv2.imshow('window',printscreenRGB)
        if cv2.waitKey(25) & 0xFF == ord('q'):
            cv2.destroyAllWindows()
            break
screen_record()
