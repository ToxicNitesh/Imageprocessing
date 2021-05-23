import numpy as np
import cv2
'''
cap = cv2.VideoCapture(0)
 
while(True):
    ret, frame = cap.read()
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    lowerYellow=np.array([ 90  ,31, 255])
    higherYellow=np.array([ 80 ,193, 205])
    mask=cv2.inRange(gray,lowerYellow,higherYellow)
    res = cv2.bitwise_and(frame,frame, mask= mask)
    cv2.imshow('original',frame)
    cv2.imshow("mask",mask)
    cv2.imshow("result",res)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()'''

green = np.uint8([[[110,50,50]]])
g=cv2.cvtColor(green, cv2.COLOR_HSV2BGR)
print(g)


cap = cv2.VideoCapture(0)

while(1):

    # Take each frame
    _, frame = cap.read()

    # Convert BGR to HSV
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    # define range of blue color in HSV
    lower_blue = np.array([110,50,50])
    upper_blue = np.array([130,255,255])

    # Threshold the HSV image to get only blue colors
    mask = cv2.inRange(hsv, lower_blue, upper_blue)

    # Bitwise-AND mask and original image
    res = cv2.bitwise_and(frame,frame, mask= mask)

    cv2.imshow('frame',frame)
    cv2.imshow('mask',mask)
    cv2.imshow('res',res)
    k = cv2.waitKey(5) & 0xFF
    if k == ord('q'):
        break

cv2.destroyAllWindows()
'''
a="'D:\cat.jpg'"
img1=cv2.imread('D:\cat.jpg',5)
cv2.imshow("img",img1)'''

