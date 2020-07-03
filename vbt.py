'''Commands:
    Setting the color mask:
        s: (save) save hsv threshold values and line length, starts main loop.
    Bucle principal:
        c: (clear) Resets all values so you can start another set.
        p: (plot) 
        q: (quit)
'''

'''for it to work you have to draw a line on the 'frame' screen, so it has a pixel-meter correlations, else it will just output NaN in the velocity'''

import cv2
import numpy as np
import time
import imutils
from matplotlib import pyplot as plt
from collections import deque

def nothing(x):
    '''Function for trackbars, literally does nothing'''
    pass

#variable init (pixel-> meters mapping)
p1 = (0,0)
p2 = (0,0)
drawing = False

def mouse_drawing(event, x, y, flags, params):
    '''Draws a line between to points for setting the scale.'''
    global drawing, p1, p2
    if event == cv2.EVENT_LBUTTONDOWN:
        if drawing is False:
            drawing = True
            p1 = (x, y)
        else:
            drawing = False
    elif event == cv2.EVENT_MOUSEMOVE:
        if drawing is True:
            p2 = (x, y)

#Trackbars
cv2.namedWindow("Tracking")
cv2.namedWindow("frame")
cv2.createTrackbar("LH", "Tracking", 25, 255, nothing)
cv2.createTrackbar("LS", "Tracking", 0, 255, nothing)
cv2.createTrackbar("LV", "Tracking", 75, 255, nothing)
cv2.createTrackbar("UH", "Tracking", 55, 255, nothing)
cv2.createTrackbar("US", "Tracking", 255, 255, nothing)
cv2.createTrackbar("UV", "Tracking", 255, 255, nothing)

#Tracking lists init
deque_length = 40
pts = deque(maxlen = deque_length)
counter = 0

#lists/variables for mean_v calculation
x_list =[]
y_list = []
t_list = []
v_x = []
v_y = []
t_v = []
scale_factor = None
mean_v_y =[]
mean_v_x = []
mean_v_t = []
vx = 0
vy = 0
v = 0
v_list = [] #list with the mean values after each rep
rep_count = 0

length_meters = 0.065  #Line length, set to diameter of tennis ball

vs = cv2.VideoCapture(0)

cv2.namedWindow("frame")
cv2.setMouseCallback("frame", mouse_drawing)

time.sleep(.5)
var = True
while var :
    '''HSV calibration and pixel/m correlation setting'''
    _, frame = vs.read()
    frame = imutils.resize(frame, width=600)

    cv2.line(frame, p1, p2, (255, 35, 35), 3)
    a = np.asarray(p1)
    b = np.asarray(p2)
    line_length = np.sqrt((b[0] - a[0])**2 + ( b[1] - a[1])**2)
    scale_factor = length_meters/(line_length)

    blurred = cv2.GaussianBlur(frame, (11,11), 0)
    hsv = cv2.cvtColor(blurred, cv2.COLOR_BGR2HSV)

    LH = cv2.getTrackbarPos("LH", "Tracking")
    LS = cv2.getTrackbarPos("LS", "Tracking")
    LV = cv2.getTrackbarPos("LV", "Tracking")
    UH = cv2.getTrackbarPos("UH", "Tracking")
    US = cv2.getTrackbarPos("US", "Tracking")
    UV = cv2.getTrackbarPos("UV", "Tracking")

    lower_hsv = np.array([LH, LS, LV])
    upper_hsv = np.array([UH, US, UV])

    mask = cv2.inRange(hsv, lower_hsv, upper_hsv)
    mask = cv2.erode(mask, None, iterations = 4)
    mask = cv2.dilate(mask, None, iterations = 4)
    res = cv2.bitwise_and(frame, frame, mask = mask)

    cv2.imshow("frame", frame)
    cv2.imshow('result', res)

    if cv2.waitKey(1) & 0xFFF == ord('s'):
        var = False
        cv2.destroyWindow("mask")
        cv2.destroyWindow("result")
        cv2.destroyWindow("Tracking")

start_time = time.time()
text_color = (255, 0, 0)
appending = False
cv2.namedWindow("v_media")

while True:
    '''Main loop, outputs mean v and changes color when reaching fatigue limit'''

    _, frame = vs.read()
    frame = imutils.resize(frame, width=600) # so it doesn't go crazy big
    blank_image = np.zeros(shape = (150, 700,3), dtype = np.uint8)
    blank_image = cv2.rectangle(blank_image, (0,0), (700,150), (255, 255, 255),-1)

    text = "v: " + str(round(v, 2)) + "m/s"
    text_reps = "reps: " + str(rep_count)

    cv2.putText(frame, text, (30,40), cv2.FONT_HERSHEY_SIMPLEX, .75, text_color, 2)
    cv2.putText(frame, text_reps, (500,40), cv2.FONT_HERSHEY_SIMPLEX, .75, text_color, 2)
    cv2.putText(blank_image, text, (30, 110), cv2.FONT_HERSHEY_SIMPLEX, 3.5, text_color, 12)

    blurred = cv2.GaussianBlur(frame, (11,11), 0)
    hsv = cv2.cvtColor(blurred, cv2.COLOR_BGR2HSV)

    mask = cv2.inRange(hsv, lower_hsv, upper_hsv)
    mask = cv2.erode(mask, None, iterations = 4)
    mask = cv2.dilate(mask, None, iterations = 4)
    res = cv2.bitwise_and(frame, frame, mask = mask)

    #Contour lines
    cnts = cv2.findContours (mask.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cnts = imutils.grab_contours(cnts) 
    center = None

    if len(cnts) > 0:
        c = max(cnts, key=cv2.contourArea)
        ((x,y), radius) = cv2.minEnclosingCircle(c)
        M = cv2.moments(c)
        center = (int(M["m10"] / M["m00"]), int(M["m01"] / M["m00"]))

        t = time.time() - start_time

        y_list.append(y)
        x_list.append(x)
        t_list.append(t)

        if len(t_list) > 2 and y_list[-1] != y_list[-2]:
            v_x.append(scale_factor *(x_list[-1] - x_list[-2])/(t_list[-1] - t_list[-2]))
            v_y.append(scale_factor *(y_list[-1] - y_list[-2])/(t_list[-1] - t_list[-2]))
            t_v.append((t_list[-1]+t_list[-2])/2)

            if v_y[-1]<0: #going up
                mean_v_y.append(v_y[-1])
                mean_v_x.append(v_x[-1])
                mean_v_t.append(t_v[-1])
                if len(mean_v_t)>2 and mean_v_t[-1] != mean_v_t[1]:
                    vx =  np.trapz(mean_v_x, mean_v_t) * (1/(mean_v_t[-1]-mean_v_t[1]))
                    vy =  np.trapz(mean_v_y, mean_v_t) * (1/(mean_v_t[-1]-mean_v_t[1]))
                    v = np.sqrt(vx**2 + vy**2)
            if v_y[-1]>0: #going down

                if len(v_list)<1 and v > 0.1 :
                    v_list.append(v)
                    rep_count += 1
                    appending = True
                elif appending == True  :
                    if v != v_list[-1] and v>0.1:
                        v_list.append(v)
                        rep_count += 1
                        #print("v_list:" + str(v_list))
                if len(v_list) > 2 and v_list[-1]<0.8*v_list[1]:
                     text_color = (0, 0, 255)
                mean_v_y =[]
                mean_v_x =[]
                mean_v_t = []
        # sets a minimum radius (pixels) helps with the noise
        if radius > 8:
            # Draw the max contour an the centroid
            cv2.circle(frame, (int(x), int(y)), int(radius), (0, 255, 255), 2)
            cv2.circle(frame, center, 5, (0, 0, 255), -1)
            pts.appendleft(center)

    # Tracked points, drawn in the video frame
    for i in np.arange(1, len(pts)):

        thickness = int(np.sqrt(deque_length / float(i + 1)) * 2.5)
        cv2.line(frame, pts[i - 1], pts[i], (0, 0, 255), thickness)

   # show the frame to our screen and increment the frame counter
    cv2.imshow("frame", frame)
    cv2.imshow("v_media", blank_image)

    key = cv2.waitKey(1) & 0xFF
    # if the 'q' key is pressed, stop the loop
    if key == ord("q"):
        break
    if key == ord("p"):
        fig, (ax1, ax2) = plt.subplots(2)
        fig.suptitle('v (t)')
        ax1.plot(t_v, v_y, 'ro',  label = 'v_y')
        ax2.plot(t_v, v_x, 'bo', label = 'v_x')
        ax2.set(xlabel = 't (s)')
        ax1.set(ylabel = 'v_y (t)')
        ax2.set(ylabel = 'v_x (t)')
        plt.show()
    if key == ord("c"):
        appending = False
        text_color = (255, 0, 0)
        rep_count = 0
        v_list = []
        x_list =[]
        y_list = []
        t_list = []
        v_x =[]
        v_y=[]
        t_v = []

vs.release()
# Close windows, releases camera
cv2.destroyAllWindows()
