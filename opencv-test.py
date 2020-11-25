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

# global variables
craters = {# list of craters drawn so far
    "x":[],
    "y":[],
    "d":[]
}
visibleCraters = [] # running list of # craters counted 


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
    print(size)
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

    # throw craters at it
    for i in range(0,500):
        craters["x"].append(random.randint(0,500))
        craters["y"].append(random.randint(0,500))
        craters["d"].append(lognormal(meanCraterSize,sigma,minCraterSize,maxCraterSize))
#        craters["d"].append(powerLaw(gamma,minCraterSize,maxCraterSize))
        
#        print(powerLaw(2,minCraterSize, maxCraterSize))
#        thisCrater = powerLaw(2,minCraterSize,maxCraterSize)        
#        print(lognormal(meanCraterSize,sigma,minCraterSize,maxCraterSize)
#        thisCrater = lognormal(meanCraterSize,sigma,minCraterSize,maxCraterSize)

        drawCrater(img, craters["x"][i],craters["y"][i], craters["d"][i])
#        drawCrater(img, random.randint(0,500), random.randint(0,500), thisCrater)
        cv2.waitKey(1)

    # plot histogram of crater sizes
    print(len(craters))
    plt.hist(craters["d"],10)
    plt.show()
    

simulate()
cv2.waitKey(0)
cv2.destroyAllWindows()
