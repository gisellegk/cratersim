import numpy as np
import cv2
import time
import random
import powerlaw
import matplotlib.pyplot as plt


#############################
######### SETTINGS  #########
#############################

# display settings
sf = 8        # scale factor
stroke = 1    # crater edge stroke

# simulation settings
avgRange = 100        # how many data points to average for saturation determination
maxCraterCount = 2000 # How long the simulation runs (10^3 years)
minCraterSize = 10    # km, diameter
maxCraterSize = 100   # km, diameter
meanCraterSize = 10   # km, diameter, for lognormal
sigma = 5             # for lognormal
gamma = 2             # for power law

# global variables
craters = {         # list of craters drawn so far
    "x":[],
    "y":[],
    "d":[]
}
visibleCraters = [] # running list of num craters counted 
runningAvg = []     # running list of averages

#############################
######### FUNCTIONS #########
#############################

# show summary of results after simulation completes
def showSummary(tsat):
    cv2.imshow('Craters: (25%)',scaleImg(redrawCanvas(round(tsat*.25)),25))
    cv2.imshow('Counted Craters: (25%)',scaleImg( countCraters(redrawCanvas(round(tsat*.25)))[1] ,25) )
    print('Number of Craters at ',round(tsat*.25),',000 years (25%): ', countCraters(redrawCanvas(round(tsat*.25)))[0])

    cv2.imshow('Craters: (50%)',scaleImg(redrawCanvas(round(tsat*.5)),25))
    cv2.imshow('Counted Craters: (50%)',scaleImg( countCraters(redrawCanvas(round(tsat*.5)))[1] ,25) )
    print('Number of Craters at ',round(tsat*.5),',000 years (50%): ', countCraters(redrawCanvas(round(tsat*.5)))[0])

    cv2.imshow('Craters: (75%)',scaleImg(redrawCanvas(round(tsat*.75)),25))
    cv2.imshow('Counted Craters: (75%)',scaleImg( countCraters(redrawCanvas(round(tsat*.75)))[1] ,25) )
    print('Number of Craters at ',round(tsat*.75),',000 years (75%): ', countCraters(redrawCanvas(round(tsat*.75)))[0])

    cv2.imshow('Craters: (100%)',scaleImg(redrawCanvas(tsat),25))
    cv2.imshow('Counted Craters: (100%)',scaleImg( countCraters(redrawCanvas(tsat))[1] ,25) )
    print('Number of Craters at ',tsat,',000 years (100%): ', countCraters(redrawCanvas(tsat))[0])

# scale display image
def scaleImg(img, sf):
    return cv2.resize(img, (int(img.shape[1]*sf/100), int(img.shape[0]*sf/100)), interpolation=cv2.INTER_AREA)

# Draws a crater on the canvas
def drawCrater(img, x, y, diameter):
    if True :#diameter < round(maxCraterSize/2+5):
        
        img = cv2.circle(img,(x*sf,y*sf), round(diameter/2*sf), (250,250,250), -1)    #white center
        img = cv2.circle(img,(x*sf,y*sf), round(diameter/2*sf), (0,0,0), round(stroke*sf))    #black crater edge
    
    #cv2.imshow('image',img)
    return img

# Redraws the canvas at a given timestamp
def redrawCanvas(time):
    img = 250 * np.ones(shape=[500*sf, 500*sf, 3], dtype=np.uint8)
    for x in range(time):
        img = drawCrater(img, craters["x"][x], craters["y"][x], craters["d"][x])
    #cv2.imshow('redrawn image', img)
    return img

# Returns a random value between maxsize and minsize, from a lognormal distribution
# https://numpy.org/doc/stable/reference/random/generated/numpy.random.lognormal.html
def lognormal(mean,sigma,minsize,maxsize):
    size = -1
    while True:
        size=np.random.lognormal(mean=np.log(mean),sigma=np.log(sigma),size=None)
        if size >= minsize and size <= maxsize:
            break
    #print(size)
    return size

# Returns a random value between maxsize and minsize, from a power law distribution
# https://mathworld.wolfram.com/RandomNumber.html
# https://stackoverflow.com/questions/17882907/python-scipy-stats-powerlaw-negative-exponent/46065079#46065079
def powerLaw(gamma, k_min, k_max):
    y=np.random.uniform(0,1)
    return ((k_max**(-gamma+1) - k_min**(-gamma+1))*y  + k_min**(-gamma+1.0))**(1.0/(-gamma + 1.0))

# Returns a count of number of craters detected in a given image.
# Performs three Hough Circle Transforms, tuned to find small, medium, and large circles in an image.
# https://docs.opencv.org/3.4/d4/d70/tutorial_hough_circle.html
def countCraters(src):
    src = cv2.cvtColor(src, cv2.COLOR_BGR2GRAY)
    src = cv2.GaussianBlur(src,(9,9),4,4)
    smallCircles = cv2.HoughCircles(src, cv2.HOUGH_GRADIENT, 1, 2*sf, param1=300, param2=90, minRadius=1, maxRadius=round(maxCraterSize*sf/8-3))
    circles = cv2.HoughCircles(src, cv2.HOUGH_GRADIENT, 1, 5*sf, param1=300, param2=100, minRadius=round(maxCraterSize*sf/8+3), maxRadius=round(maxCraterSize*sf/4)-3)
    bigCircles = cv2.HoughCircles(src, cv2.HOUGH_GRADIENT, 2, 30*sf, param1=100, param2=150, minRadius=round(maxCraterSize*sf/4+3), maxRadius=round(maxCraterSize/2*sf))
    src = cv2.cvtColor(src, cv2.COLOR_GRAY2BGR)
    thisCraterCount = 0
    
    if smallCircles is not None:
        smallCircles = np.uint16(np.around(smallCircles))
        thisCraterCount += len(smallCircles[0, :])
        for k in smallCircles[0, :]:
            center = (k[0], k[1])
            # circle center
            cv2.circle(src, center, 1, (0,0,0), 3)
            # circle outline
            radius = k[2]
            cv2.circle(src, center, radius, (0,250,0), 4)                                
    if circles is not None:
        circles = np.uint16(np.around(circles))
        thisCraterCount += len(circles[0, :])
        for k in circles[0, :]:
            center = (k[0], k[1])
            # circle center
            cv2.circle(src, center, 1, (0,0,0), 3)
            # circle outline
            radius = k[2]
            cv2.circle(src, center, radius, (0,0,250), 4)
    if bigCircles is not None:
        bigCircles = np.uint16(np.around(bigCircles))
        thisCraterCount += len(bigCircles[0, :])
        for k in bigCircles[0, :]:
            center = (k[0], k[1])
            # circle center
            cv2.circle(src, center, 1, (250,0,250), 3)
            # circle outline
            radius = k[2]
            cv2.circle(src,center,radius,(250,0,250),4)
    return (thisCraterCount,src)


#############################
### SIMULATION  AND PLOTS ###
#############################
def simulate():
    #create blank image
    img = 250 * np.ones(shape=[500*sf, 500*sf, 3], dtype=np.uint8)
    i = 0
    # throw craters at it
    while True:
        # create a new crater:
        craters["x"].append(random.randint(0,500))
        craters["y"].append(random.randint(0,500))
        #craters["d"].append(lognormal(meanCraterSize,sigma,minCraterSize,maxCraterSize))
        craters["d"].append(powerLaw(gamma,minCraterSize,maxCraterSize))

        # draw crater on canvas
        print(i,", ",craters["x"][i],", ",craters["y"][i],", ", craters["d"][i])
        drawCrater(img, craters["x"][i],craters["y"][i], craters["d"][i])

        ### count visible craters
        
        src = img.copy()
        thisCraterCount,src = countCraters(src)
        visibleCraters.append(thisCraterCount);
        
        # calculate running average
        if len(visibleCraters) >= avgRange:
            avg = 0
            for j in range(0,avgRange):
                avg += visibleCraters[-avgRange+j] # sum last n craters
            avg = avg / avgRange
            runningAvg.append(avg)
        elif len(visibleCraters) >= avgRange/2: 
            runningAvg.append(None) # time adjustment for graphing

        if len(craters["d"]) >= maxCraterCount:             
            break
        
        if i%100 == 0:
            src = cv2.resize(src, (int(src.shape[1]*25/100), int(src.shape[0]*25/100)), interpolation=cv2.INTER_AREA)
            cv2.imshow("detected circles", src)
            cv2.waitKey(1) # update image
        i = i+1

    # plot histogram of crater sizes
    plt.subplot(121)
    plt.title("Histogram of Crater Sizes:\nPower Law Distribution")
    plt.ylabel("Number of Craters")
    plt.xlabel("Crater Diameter (km) (bucket size=10)")
    plt.hist(craters["d"],10)

    # plot # visible craters over time  (part c)
    plt.subplot(122)
    plt.title("Total Visible Craters over Time:\nPower Law Distribution")
    plt.ylabel("Number of Craters")
    plt.xlabel("Time (10^3 years)")
    plt.plot(visibleCraters)
    plt.plot(runningAvg)
    
    plt.show()


simulate()
cv2.waitKey(0)
cv2.destroyAllWindows()
