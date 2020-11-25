import numpy as np
import cv2
import time
import random

sf = 2        # scale factor
stroke = 1    # crater edge stroke


def drawCrater(img, x, y, diameter):
    img = cv2.circle(img,(x*sf,y*sf), round(diameter/2*sf), 255, -1)    #white center
    img = cv2.circle(img,(x*sf,y*sf), round(diameter/2*sf), 0, 1*sf)    #black crater edge
    
    cv2.imshow('image',img)


img = 200 * np.ones(shape=[500*sf, 500*sf, 1], dtype=np.uint8)
#drawCrater(img, 100, 100, 10)
#drawCrater(img, 120, 120, 10)
#drawCrater(img, 110, 100, 30)

for x in range(0,100):
    drawCrater(img, random.randint(0,500), random.randint(0,500), random.randint(10,50))
    cv2.waitKey(100) 


cv2.waitKey(0)


cv2.destroyAllWindows()
