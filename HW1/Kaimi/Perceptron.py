import numpy as np
import matplotlib.pyplot as plt
import random
from scipy.interpolate import interp1d


## -- Preparing Canvas
plt.clf()

fig = plt.figure()
ax = fig.add_subplot(111)

ax.annotate("Class A", (-4,10), xytext=(-4,11))
ax.annotate("Class B", (10,0), xytext=(11,0))

plt.grid()
plt.xlim(-10,10)
plt.ylim(-10,10)

plt.draw()


    ## -- Activation Function
def sign(n):
    '''
    Input: n -- a single float value
    output: the sign of the input
    '''
    if n >= 0:
        return 1
    else:
        return -1
    
    ## -- Arbitrary linear function to classify points
def f(x):
    # y = mx + b
    X = np.array(x)
    return 3.0 * x + 2.0

class Point():
    global x
    global y
    global label
    
    def __init__(self):
        '''
        Input: N/A
        Output: a random point with x,y coords and "label"
        '''
        self.x = random.uniform(-10,10)
        self.y = random.uniform(-10,10)
        
        lineY = f(self.x)
        
        if (self.y >= lineY):
            self.label = 1
        else:
            self.label = -1
        #'''  
        #print self.x
        #print self.y
        #print label
        #'''

    def show(self):
        '''
        Input: N/A
        Output: - Plot open circles for class A
                - Plot closed circles for class B
                - Plot a line of the "cut off"
        '''
        
        ax.plot([-15,15], [f(-15), f(15)], color="grey")
        
        if self.label == 1:
            ax.scatter(self.x,self.y, color="k", s=60)
        else:
            ax.scatter(self.x,self.y, facecolors='none', 
            edgecolors="k", s=60)
            

class Perceptron():
    global weights
    global lr
    global bias ## - in the case input is (0,0)
    
    weights = np.zeros(2) ##Each input (x,y) starting weights
    lr = 2 ## the learning weight
    
    ## -- Constructor
    def __init__(self):
        '''
        Input: N/A
        Output: 2 random starting weights for each input (x,y)
        '''
        
        for i in xrange(len(weights)):
            weights[i] = random.uniform(-1,1)
            
    def guess_func(self, inputs):        
        '''
        Input: Array of points
        Output: Sum inputs*weights
        '''
        
        Sum = 0

        for i in xrange(len(weights)):
            Sum += np.array(inputs[i])*np.array(weights[i])
            ##print weights[i]
        
        output = sign(Sum)
        
        return output
    
    def train(self, inputs, target):  
        '''
        Supervised learning function to "tune" the weights
        
        Input:
            - inputs -- array of points
            - target -- each points classification (+/- 1)
        Output:
            - Adjust weights depending on error and learning rate
            
            Formula:
                New Weights = Error * inputs * learning rate
        '''
        
        Guess = self.guess_func(inputs)        
        
        error = target - Guess
        
        for i in xrange(len(weights)):
            weights[i] += error*inputs[i]*lr
            
            ##print("Inputs: %s")%inputs[i]
            ##print("Weights: %s")%(weights[i])          

def main():
    
    brain = Perceptron()

    points = []
    
    ## -- Generate 100 random points using the Point() class
    for i in xrange(100):
        pt = Point()
        points.append(pt)
    
    for pt in points:
        pt.show()
        
        inputs = np.array([pt.x, pt.y])
        target = pt.label

        
        brain.train(inputs, target)
        
        guess = brain.guess_func(inputs)
        
        
        if guess == target:
            ax.plot(pt.x, pt.y, "c.", ms=10)
        else:
            ax.plot(pt.x, pt.y, "r.", ms=10)
        

if __name__ == "__main__":
    main()
    plt.show()
