# -*- coding: utf-8 -*-
"""
Created on Fri May  1 12:12:32 2020

@author: krist
"""


import numpy as np
import cv2

img = np.zeros([700, 700,3], np.uint8)  #y, x image
img = cv2.rectangle(img, (0,0), (700,700), (51,51,51),-1)

position = (350, 700.5)
orientation = (0,-1)

saved_orientation = []
saved_position = []

angle = 25 * np.pi/180

n = 6#Number of iterations for the L-System
length = 180
string = "+X"
aux = ""

#This generates the phrase
def rules (string, aux):
    for i in range(len(string)):
        if string[i] == "F":
            aux = aux + "FF"
        elif string[i] =="X":
            aux = aux + "F+[[X]-X]-F[-FX]+X"
        else:
            aux = aux + string[i]
    string = aux
    return string

def rotate (orientation, angle):
    R = np.array([[np.cos(angle), -np.sin(angle)],
                  [+np.sin(angle), np.cos(angle)]])
    orientation = R.dot(orientation)
    return orientation

for j in range(n):
    length = length *0.5
    string = rules(string, aux)
    
    aux = ""


for i in range(len(string)):
    if string[i] == "F":
        end = position + length * np.array(orientation)
        end = (int(end[0]), int(end[1]))
        position = (int(position[0]), int(position[1]))
        cv2.line(img, position, end, (255,255,255),1)
        position = end
    elif string[i] == "+":
        orientation = rotate(orientation, +angle)
    elif string[i] == "-":
        orientation = rotate(orientation, -angle)
    elif string[i] == "[":
        saved_position.append(position)
        saved_orientation.append(orientation)
    elif string[i] == "]":
        position = saved_position[-1]
        orientation = saved_orientation[-1]
        saved_position.pop(-1)
        saved_orientation.pop(-1)

cv2.imshow('img', img)

cv2.waitKey(0)
cv2.destroyAllWindows()