import win32api as wapi
import time

keyList = ["\b"]
for char in "ABCDEFGHIJKLMNOPQRSTUVWXYZ 123456789,.'APS$/\\":
    keyList.append(char)

def key_check():
    keys = []
    for key in keyList:
        if wapi.GetAsyncKeyState(ord(key)):
            keys.append(key)
    if 'A' in keys:
        return 'A'
    elif 'B' in keys:
        return 'B'
    elif 'C' in keys:
        return 'C'
    elif 'D' in keys:
        return 'D'
    elif ' ' in keys:
        return ' '
    elif 'E' in keys:
        return 'E'
    elif 'F' in keys:
        return 'F'
    elif 'G' in keys:
        return 'G'
    elif 'H' in keys:
        return 'H'
    elif 'I' in keys:
        return 'I'
    elif 'J' in keys:
        return 'J'
    elif 'K' in keys:
        return 'K'
    elif 'L' in keys:
        return 'L'
    elif 'M' in keys:
        return 'M'
    elif 'O' in keys:
        return 'O'
    elif 'P' in keys:
        return 'P'
    elif 'Q' in keys:
        return 'Q'
    elif 'R' in keys:
        return 'R'
    elif 'S' in keys:
        return 'S'
    '''
    elif 'G' in keys:
        return 'G'
    elif 'G' in keys:
        return 'G'
    elif 'G' in keys:
        return 'G'
    elif 'G' in keys:
        return 'G'
    elif 'G' in keys:
        return 'G'
    elif 'G' in keys:
        return 'G'
    elif 'G' in keys:
        return 'G'
    elif 'G' in keys:
        return 'G'
    elif 'G' in keys:
        return 'G'
    elif 'G' in keys:
        return 'G'
    elif 'G' in keys:
        return 'G'
    '''
