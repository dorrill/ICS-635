#Anirvan's perceptron. First Attempt. Does not create images for animation

import sys
import os
import numpy as np
from numpy import random as rad
from numpy import array

import matplotlib.pyplot as plt

min = 0
max = 10

## -- Preparing Canvas
plt.clf()
fig = plt.figure()
ax = fig.add_subplot(111)
axes = plt.gca()
axes.set_xlim([0,10])
axes.set_ylim([0,10])
plt.draw()

def graph_expected(formula, min, max):
    x = np.linspace(min, max, 200)
    y = eval(formula)
    plt.plot(x,y, color = "black",linestyle = 'dashed')

def graph_LEARNED(formula, min, max):
    x = np.linspace(min, max, 200)
    y = eval(formula)
    plt.plot(x,y, color = "purple",linestyle = 'solid')   

def main():
        N = 9 #number of training points in data set
        x = [6, 1, 3, 2, 5, 6, 7, 8, 9] #training data set
        y = [9, 6, 4, 5, 1, 4, 3, 5, 7]
        label = [-1, -1, -1, -1, 1, 1, 1, 1, 1]; #labels for data, or teaching set

        for i in range(0,N):
            if label[i] == 1:
                ax.plot(x[i], y[i], "c.",  marker ='x', ms=10)
            else:
                ax.plot(x[i], y[i], "c.",  marker ='o', ms=10)

        m = [0.1, 1] #m are my weigts
        learning_rate = 0.01
        learning_rate_b = 0.0002
        max_attempts = 10

        graph_expected("x*{} + {}".format(m[0],m[1]), min, max) #plot initial guess line

        for i in range(0,N):
            attempts = 1
            classification = 0 #initialize

            while classification == 0 and attempts < max_attempts:
                y_calc = m[0]*x[i]+m[1]

                if y_calc < y[i]:
                    perceptron_output = -1
                else:
                    perceptron_output = 1

                if perceptron_output*label[i] == 1:
                    classification = 1 #good classification
                else:
                    classification = 0 #bad classification

                if classification == 0:
                    m[0] = m[0] + learning_rate*(label[i]-perceptron_output)*x[i]
                    m[1] = m[1] + learning_rate_b*(label[i]-perceptron_output)
                    graph_LEARNED("x*{} + {}".format(m[0] ,m[1]), min, max)
                attempts = attempts + 1
                if attempts == max_attempts:
                    print("I give up with point",i)

        #mag_w = np.sqrt((dotproduct(w,w,3)))

        print (m)

main()
#plt.show()
fig = plt.figure(2)
fig.savefig('Plots/1.png', bbox_inches='tight')
