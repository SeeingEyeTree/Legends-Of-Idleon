import numpy as np
import pyautogui
import time
from Keys import PressKey, ReleaseKey , I, ESCAPE, C, M, DELETE, FOUR, THREE, Q, FIVE, ZERO, ENTER, ONE
import Keys
from grab_screen import grab_screen
import cv2
from GetKeys import key_check
from BasicFunctions import click, open_chest_page, press, hold_down,inventory_indexer

NUMBER_OF_MEALS = 67
#start_on_key('S') 
key_codes = {
    "0": Keys.ZERO,
    "1": Keys.ONE,
    "2": Keys.TWO,
    "3": Keys.THREE,
    "4": Keys.FOUR,
    "5": Keys.FIVE,
    "6": Keys.SIX,
    "7": Keys.SEVEN,
    "8": Keys.EIGHT,
    "9": Keys.NINE
}

def match_templat(where,what,threshold):
    res = cv2.matchTemplate(where,what,cv2.TM_CCOEFF_NORMED)
    loc = np.where( res >= threshold)
    '''
    whereRGB=cv2.cvtColor(where,cv2.COLOR_GRAY2RGB)
    w1, h1 = what.shape[::-1]
    for pt1 in zip(*loc[::-1]):
            cv2.rectangle(whereRGB, pt1, (pt1[0] + w1, pt1[1] + h1), (0,0,255), 2)
    cv2.imshow('matches',whereRGB)
    print(loc)
    '''    
    return loc 


def inventory(page, slotx, sloty):
    in_page = page
    page = 0
    x = 1287
    y=233
    while page != in_page:
        x+=116
        page+=1
    click(x,y)
    slotx=(slotx*120)+1320
    sloty=(sloty*120)+360
    pyautogui.moveTo(slotx,sloty)
    
    
    
def vials():
    print('Will you rember that you need to start with your inventory open this time? ')
    x=0
    looptimes=input('number of times to loop')
    looptimes=int(looptimes)
    time.sleep(2)
    while x<looptimes:
        pyautogui.moveTo(1320, 360)
        pyautogui.dragTo(280, 450, 0.3, button='left')
        time.sleep(3)
        press(ESCAPE)
        click(730,780)
        press(I)
        time.sleep(2)
        x+=1
       
       
def buy_slot(hold_down_t, slot):
    sloty=(slot*130)+390
    hold_down(hold_down_t,1047,sloty)
    click(689,sloty)
    

def click_claim_and_col():
    screen=grab_screen()
    screen_gray= cv2.cvtColor(screen, cv2.COLOR_BGR2GRAY)
    claim=cv2.imread('./ClaimButon.PNG')
    claim_gray= cv2.cvtColor(claim, cv2.COLOR_BGR2GRAY)
    y,x=match_templat(screen_gray,claim_gray,0.9)
    y=y[0]+50
    x=x[0]+50
    click(x,y)
    time.sleep(2)
    x,y=find_player()
    y-=50
    pyautogui.moveTo(120,y)
    pyautogui.mouseDown()
    pyautogui.moveTo(1800,y,4)
    pyautogui.mouseUp()


def click_claim_and_col():
    screen=grab_screen()
    screen_gray= cv2.cvtColor(screen, cv2.COLOR_BGR2GRAY)
    claim=cv2.imread('C./troll_re.PNG')
    claim_gray= cv2.cvtColor(claim, cv2.COLOR_BGR2GRAY)
    y,x=match_templat(screen_gray,claim_gray,0.9)
    y=y[0]+50
    x=x[0]+50
    click(x,y)
    time.sleep(2)
    x,y=find_player()
    y-=50
    pyautogui.moveTo(120,y)
    pyautogui.mouseDown()
    pyautogui.moveTo(1800,y,4)
    pyautogui.mouseUp()
    
    
    
def anvil_collection(Startx,Starty,page):
    if page==0:
        press(ESCAPE)
        click(1686,641)
        click(607,765)
    if page==1:
        press(ESCAPE)
        click(1686,641)
        click(1403,767)
    click(Startx,Starty)
    click(1000,760)
    time.sleep(2.5)
    try:
        click_claim_and_col()
    except:
        print('No claim button')
    press(C)
    click(1170,225)
    click(900,400)
    click(1030,230)
    click(890,220)
    press(ESCAPE)
    
    
def anvil():
    loop=0
    startx=666
    starty=360
    while loop<3:
        anvil_collection(startx,starty,0)
        startx+=334
        loop+=1
    loop=0
    startx=666
    starty=666
    while loop<3:
        anvil_collection(startx,starty,0)
        startx+=334
        loop+=1
    time.sleep(0.2)
    loop=0
    startx=666
    starty=360
    while loop<3:
        anvil_collection(startx,starty,1)
        startx+=334
        loop+=1
        
        
def shards_and_tps():
    click(1137,800)
    click(657,855)
    hold_down(2.5,1047,603)
    hold_down(1,1047,726)
    click(687,783)
    click(687,639)
    
    
def active_col():
    x,y=find_player()
    x-=130
    y-=50
    pyautogui.moveTo(x,y)
    pyautogui.mouseDown()
    pyautogui.move(500,1,1.5)
    pyautogui.mouseUp()
    

def get_ladles(numladles):
    open_chest_page()
    click(228, 377) # click quick tap so split stack is alwas on after sequens
    click(241, 730) # click slpit stack
    for i in range(NUMBER_OF_MEALS):
        click(524,520,0.01)
        PressKey(key_codes[numladles])
        PressKey(ENTER)
    press(Keys.C)
    press(Keys.C)


def cook(numladles):
    get_ladles(numladles)
    click(890,785) #clicks first kitchen
    click(1662,318)  # click go back
    selfs = 4
    for i in range(selfs):
        click(1566,1084) # clicks shelf left
    press(Keys.C)
    press(Keys.C)
    
    ladle_slots = list = [i for i in range(NUMBER_OF_MEALS)]
    menu_offset = 180;
    flip = 0
    for shelf in range(3):
        y_meal = 435
        for row in range(4):
            ladle_slots = meal_row(y_meal, ladle_slots, menu_offset * flip)
            flip = 1 - flip
            y_meal += 155
            click(1498,879) # click menu
            pyautogui.scroll(-10)
            pyautogui.scroll(-10)
            if flip == 0:
                pyautogui.scroll(-10)
            press(Keys.C)
            press(Keys.C)
        click(890,785) #clicks first kitchen
        click(1662,318)  # click go back
        click(1784,1087) # shelf right
        press(Keys.C)
        press(Keys.C)

        

def meal_row(y_meal, ladle_slots, menu_offset):
    x_meal = 1329
    y_menu = 281
    for i in range(5):
        click(890,785) #clicks first kitchen
        click(1662,318) #click go back
        click(x_meal,y_meal)
        click(1706,1036)  #click cook all
        press(Keys.I)
        inventory_indexer(str(ladle_slots[0]))
        ladle_slots.pop(0)
        hold_down()
        press(Keys.I)
        click(1498,879) # click menu
        for i in range(10): # if you are getting more than 10 levels you negleted cooking wayyy to long
            click(1722, (y_menu + menu_offset)) # click meal
        x_meal += 190
        y_menu += 180
        press(Keys.C)
        press(Keys.C)
    return ladle_slots


def boss(order):
    for i in order:
        if i == "1":
            click(1642,881)
        elif i == "2":
            click(1642,1023)
        else:
            click(1663,1193)
    
    
if __name__ == "__main__":
    time.sleep(5)
    cook("3")
    #boss("2232213332213")
    #print(inventory_indexer)

    #pyautogui.scroll(-10)
    #menu
    #click(1722,284)
    #click(1721,458)
    #458 - 284

# inv
    #click(1780,522)
    #click(1997,521)i