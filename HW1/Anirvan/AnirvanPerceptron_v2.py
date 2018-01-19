#Anirvan's perceptron. Improves v1, and adds creation of images for making a gif

import sys
import os
import numpy as np
from numpy import random as rad
from numpy import array

import matplotlib.pyplot as plt

min = 0
max = 10

def graph_expected(formula, min, max):
    x = np.linspace(min, max, 200)
    y = eval(formula)
    plt.plot(x,y, color = "black",linestyle = 'dashed')

def graph_LEARNED(formula, min, max):
    x = np.linspace(min, max, 200)
    y = eval(formula)
    plt.plot(x,y, color = "purple",linestyle = 'solid')

def plots_for_gif(N,x,y,m,label,counter1, counter2):
    fig = plt.figure()
    ax = fig.add_subplot(111)

    axes = plt.gca()
    plt.xlabel('x')
    plt.ylabel('y')
    plt.title('Slope = %s, Intercept = %s'%(m[0], m[1]))
    axes.set_xlim([0,10])
    axes.set_ylim([0,10])
    
    for i in range(0,N):
        if label[i] == 1:
            ax.plot(x[i], y[i], "c.",  marker ='x', ms=10)
        else:
            ax.plot(x[i], y[i], "c.",  marker ='o', ms=10)

    graph_LEARNED("x*{} + {}".format(m[0] ,m[1]), min, max)
    plt.savefig('Plots/Anim_i_%s_a_%s.png'%(counter1, counter2), bbox_inches='tight' )    

def main():
        N = 9 #number of training points in data set
        # x = [6, 1, 3, 2, 5, 6, 7, 8, 9] #training data set
        # y = [9, 6, 4, 5, 1, 4, 3, 5, 7]
        # label = [-1, -1, -1, -1, 1, 1, 1, 1, 1] #labels for data, or teaching set

        x = [1, 2, 3, 5, 6, 6, 7, 8, 9] #training data set
        y = [6, 5, 4, 1, 9, 4, 3, 5, 7]
        label = [1, 1, 1, 1, -1, -1, -1, -1, -1]


        m = [0.01, 1] #m are my weigts, [0.01, 1], [-4, 8]
        learning_rate = 0.15 #0.005
        learning_rate_b = 1 #0.0002
        max_attempts = 200 #20

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
                    plots_for_gif(N,x,y,m,label,i,attempts)
                
                print i, attempts, classification
                attempts = attempts + 1
                if attempts == max_attempts:
                    print("I give up with point",i)

        #mag_w = np.sqrt((dotproduct(w,w,3)))

        print (m)

main()
#plt.show()
fig = plt.figure(2)

