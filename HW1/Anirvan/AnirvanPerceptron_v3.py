#Anirvan's perceptron model, based on v2. Now perceptron keep learning over the data set till error
#for each point is zero

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

def plots_for_gif(N,x,y,m,label,counter0,counter1, counter2):
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
    plt.savefig('Plots/Anim_run_%s_i_%s_a_%s.png'%(counter0, counter1, counter2), bbox_inches='tight' )

def classify_point(x, y, m, label):
	y_calc = m[0]*x+m[1]
	if y_calc < y:
		perceptron_output = -1
	else:
		perceptron_output = 1

	if perceptron_output*label ==1:
		classification = 0 #good classification
	else:
		classification = 1 #bad classification
	return classification

def main():
        # N = 9 #number of training points in data set
        # x = [6, 1, 3, 2, 5, 6, 7, 8, 9] #training data set
        # y = [9, 6, 4, 5, 1, 4, 3, 5, 7]
        # label = [-1, -1, -1, -1, 1, 1, 1, 1, 1] #labels for data, or teaching set

        N = 9
        x = [1, 2, 3, 5, 6, 6, 7, 8, 9] #training data set no. 2
        y = [6, 5, 4, 1, 9, 4, 3, 5, 7]
        label = [1, 1, 1, 1, -1, -1, -1, -1, -1]

        # N = 4
        # x = [2, 3, 7, 9] #training data set no. 3, showing XOR
        # y = [2, 8, 2, 8]
        # label = [1, -1, -1, 1]


        m = [0.01, 1] #m are my weigts, [0.01, 1], [-4, 8]
        learning_rate = 0.15 #0.005
        learning_rate_b = 1 #0.0002
        max_attempts = 200 #20
        total_classification_errors = 1 #initial guess
        epoch = 1

        plots_for_gif(N, x, y, m, label, 0, 0, 0)

        while total_classification_errors > 0:
	        for i in range(0,N):
	            attempts = 1
	            classification = 1 #initialize with a bad classication

	            while classification == 1 and attempts < max_attempts:
	                classification = classify_point(x[i], y[i], m, label[i]) #classify point x[i], y[i]

	                if classification == 1: #if classification is bad, weights are shifted/trained.
	                	perceptron_output = -1*label[i] #technically it should be -1/label[i], but since labels are -1 or +1, it ends up being equivalent.
	                	m[0] = m[0] + learning_rate*(label[i]-perceptron_output)*x[i]
	                	m[1] = m[1] + learning_rate_b*(label[i]-perceptron_output)
	                	plots_for_gif(N, x, y, m, label, epoch, i, attempts)
	                
	                print i, attempts, classification
	                attempts = attempts + 1
	                if attempts == max_attempts:
	                    print("I give up with point",i)

			#now test the final weights on the sample, to check if they still classify all training points correctly
			#for each point in sample, see if it still correctly classified.
			#if any classification is bad, repeat the for loop
			total_classification_errors = 0
			for i in range(0,N):
				classification = classify_point(x[i], y[i], m, label[i])
				total_classification_errors = total_classification_errors + classification

			print epoch, total_classification_errors
			epoch = epoch + 1

        print (m)

main()
#plt.show()
fig = plt.figure(2)

