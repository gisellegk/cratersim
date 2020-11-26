import numpy as np
import cv2
import time
import random
import powerlaw
import matplotlib.pyplot as plt


# display settings
sf = 2        # scale factor
stroke = 1    # crater edge stroke

# simulation settings
minCraterSize = 10  #km, diameter
maxCraterSize = 100 #km, diameter
meanCraterSize = 10 #km, diameter, for lognormal
sigma = 5           # for lognormal
gamma = 2           # for power law
avgRange = 10       # how many data points to average for saturation determination

# global variables
craters = {         # list of craters drawn so far
    "x":[],
    "y":[],
    "d":[]
}
visibleCraters = [] # running list of num craters counted 
runningAvg = []     # running list of averages

def drawCrater(img, x, y, diameter):
    img = cv2.circle(img,(x*sf,y*sf), round(diameter/2*sf), 255, -1)    #white center
    img = cv2.circle(img,(x*sf,y*sf), round(diameter/2*sf), 0, 1*sf)    #black crater edge
    
    cv2.imshow('image',img)

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

def simulate():
    #create blank image
    img = 200 * np.ones(shape=[500*sf, 500*sf, 1], dtype=np.uint8)
    i = 0
    # throw craters at it
    while True:
        # create a new crater:
        craters["x"].append(random.randint(0,500))
        craters["y"].append(random.randint(0,500))
        #craters["d"].append(lognormal(meanCraterSize,sigma,minCraterSize,maxCraterSize))
        craters["d"].append(powerLaw(gamma,minCraterSize,maxCraterSize))

        # draw crater on canvas
        drawCrater(img, craters["x"][i],craters["y"][i], craters["d"][i])

        ### TODO: count visible craters
        visibleCraters.append(i);
        
        # calculate running average
        if len(visibleCraters) >= avgRange:
            avg = 0
            for j in range(0,avgRange):
                avg += visibleCraters[-avgRange+j] # sum last n craters
            avg = avg / avgRange
            runningAvg.append(avg)

        if len(visibleCraters) >= 2000: ### TODO: determine saturation to break out of the loop
            
            break

        i = i+1

#        cv2.waitKey(1)
    print('Number of Visible Craters: ', visibleCraters[i])
    print('Running Average: ', runningAvg[-1])
    print('Year:', 1000*len(visibleCraters))

    # plot histogram of crater sizes
    ### TODO: labels
    print(len(craters["x"]))
    plt.subplot(131)
    plt.hist(craters["d"],10)

    # plot # visible craters over time  (part c)
    ### TODO: labels, mark saturation
    plt.subplot(132)
    plt.plot(visibleCraters)
    
    # plot running average over time
    ### TODO: Labels, tune saturation calculation
    plt.subplot(133)
    plt.plot(runningAvg)
    
    plt.show()

    ### TODO: Plot 25%, 50%, 75%


simulate()
cv2.waitKey(0)
cv2.destroyAllWindows()
