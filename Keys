#Not done by me

# direct inputs
# source to this solution and code:
# http://stackoverflow.com/questions/14489013/simulate-python-keypresses-for-controlling-a-game
# http://www.flint.jp/misc/?q=dik&lang=en

import ctypes
import time

SendInput = ctypes.windll.user32.SendInput

# Numbers
ZERO = 0x0B
ONE = 0x02
TWO = 0x03
THREE= 0x04
FOUR=0x05
FIVE=0x06
SIX=0x07
SEVEN =0x08
EIGHT=0x09
NINE=0x0A

# Misc
SPACE = 0x39
ESCAPE = 0x01
MINUS = 0x0C  # -
EQUALS = 0x0D # =
BACK = 0x0E   
TAB = 0x0F  
ENTER = 0x1C
DELETE = 0xD3
LCTRL = 0x1D 
LSHIFT= 0x2A
LALT=0x38
CAPLOCK=0x3A
F4=0x3E

#Letters
A = 0x1E
B = 0x30
C = 0x2E
D = 0x20
E = 0x12
F = 0x21
G = 0x22
H = 0x23
I = 0x17
J = 0x24
K = 0x25
L = 0x26
M = 0x32
N = 0x31
O = 0x18
P = 0x19
Q = 0x10
R = 0x13
S = 0x1F
T = 0x14
U = 0x16
V = 0x2F
W = 0x11
X = 0x2D
Y = 0x15
Z = 0x2C

  
# C struct redefinitions 
PUL = ctypes.POINTER(ctypes.c_ulong)
class KeyBdInput(ctypes.Structure):
    _fields_ = [("wVk", ctypes.c_ushort),
                ("wScan", ctypes.c_ushort),
                ("dwFlags", ctypes.c_ulong),
                ("time", ctypes.c_ulong),
                ("dwExtraInfo", PUL)]

class HardwareInput(ctypes.Structure):
    _fields_ = [("uMsg", ctypes.c_ulong),
                ("wParamL", ctypes.c_short),
                ("wParamH", ctypes.c_ushort)]

class MouseInput(ctypes.Structure):
    _fields_ = [("dx", ctypes.c_long),
                ("dy", ctypes.c_long),
                ("mouseData", ctypes.c_ulong),
                ("dwFlags", ctypes.c_ulong),
                ("time",ctypes.c_ulong),
                ("dwExtraInfo", PUL)]

class Input_I(ctypes.Union):
    _fields_ = [("ki", KeyBdInput),
                 ("mi", MouseInput),
                 ("hi", HardwareInput)]

class Input(ctypes.Structure):
    _fields_ = [("type", ctypes.c_ulong),
                ("ii", Input_I)]

# Actuals Functions

def PressKey(hexKeyCode):
    extra = ctypes.c_ulong(0)
    ii_ = Input_I()
    ii_.ki = KeyBdInput( 0, hexKeyCode, 0x0008, 0, ctypes.pointer(extra) )
    x = Input( ctypes.c_ulong(1), ii_ )
    ctypes.windll.user32.SendInput(1, ctypes.pointer(x), ctypes.sizeof(x))

def ReleaseKey(hexKeyCode):
    extra = ctypes.c_ulong(0)
    ii_ = Input_I()
    ii_.ki = KeyBdInput( 0, hexKeyCode, 0x0008 | 0x0002, 0, ctypes.pointer(extra) )
    x = Input( ctypes.c_ulong(1), ii_ )
    ctypes.windll.user32.SendInput(1, ctypes.pointer(x), ctypes.sizeof(x))

if __name__ == '__main__':
    PressKey(0x11)
    time.sleep(1)
    ReleaseKey(0x11)
    time.sleep(1)
