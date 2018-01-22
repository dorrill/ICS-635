#Anirvan's perceptron
#Creates a random true line, and generates data points with labels about that line
#Then, it uses the perceptron model to try to learn the true line

# Variables to play with
# N, number of data points in sample
# m, my weigts; random guess
# learning_rate; for parameter m0 or slope; a random guess
# learning_rate_b; for parameter m1 or y-intercept; a random guess
# max_epochs; number of runs allowed over data set to learn true line


import sys
import os, subprocess
import random
import numpy as np
from numpy import array
import math

import matplotlib.pyplot as plt

#define overall bounds for data, and parameters for the true line, which the perceptron needs to learn
min = 0
max = 1

def draw_line(formula, min, max, color = False):
    x = np.linspace(min, max, 200)
    y = eval(formula)
    if color == False:
    	plt.plot(x,y, color = "black",linestyle = 'dashed')
    else:
    	plt.plot(x,y, color = "purple",linestyle = 'solid')


def plots_for_gif(N, learning_rate, x, y, m, m_true, label, counter0, counter1, counter2, true_line = False):
    fig = plt.figure(figsize=(8,8))
    ax = fig.add_subplot(111)

    axes = plt.gca()
    plt.xlabel('x')
    plt.ylabel('y')
    plt.title('N: %d, LR: %s, Iterations: %d, True slope: %.3f, Learnt slope = %.3f, True intercept = %.3f, Learnt intercept = %.3f'%(N, learning_rate, counter2, m_true[0], m[0], m_true[1], m[1]))
    axes.set_xlim([0,1])
    axes.set_ylim([0,1])
    
    for i in range(0,N):
        if label[i] == 1:
            ax.plot(x[i], y[i], "c.",  marker ='x', ms=5)
        else:
            ax.plot(x[i], y[i], "c.",  marker ='o', ms=5)
    if true_line == False:
    	draw_line("x*{} + {}".format(m[0] ,m[1]), min, max, True)
    	plt.savefig('Plots/Anim_N_%s_run_%s_i_%s_iter_%s.png'%(N, counter0, counter1, counter2), bbox_inches='tight' )
    else:
    	draw_line("x*{} + {}".format(m[0] ,m[1]), min, max, True)
    	draw_line("x*{} + {}".format(m_true[0] ,m_true[1]), min, max, False)
    	plt.savefig('Plots/Anim_N_%s_LR_%s_run_%s_i_%s_iter_%s.png'%(N, learning_rate, counter0, counter1, counter2), bbox_inches='tight' )
    

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

def find_margin(x, y, m): #for finding theoretical max_iterations = R^2/margin^2,
	possible_margin = abs(x*(-m[0])+y-m[1])/math.sqrt(m[0]*m[0]+1) #distance between the true line and the closest point
	return possible_margin

def find_Radius(x, y, x0, y0): #for finding theoretical max_iterations = R^2/margin^2,
	possible_R = math.sqrt( (x-x0)*(x-x0)+(y-y0)*(y-y0) ) #furthest point from the center of canvas: (0.5, 0.5)
	return possible_R

class Sample_data_generator:
    def __init__(self, N):
        # Random linearly separated data
        xA,yA,xB,yB = [random.uniform(0.25*max,0.75*max) for i in range(4)] #ensures true line is near the center of the canvas
        self.V = np.array([xB*yA-xA*yB, yB-yA, xA-xB])
        self.X = self.generate_points(N)
 
    def generate_points(self, N):
        X = []
        for i in range(N):
            x1,x2 = [random.uniform(min, max) for i in range(2)]
            x = np.array([1,x1,x2])
            s = int(np.sign(self.V.T.dot(x))) #generate labels
            X.append((x, s, self.V))
        return X

def main():

        # N = 10
        # x = [.1, .2, .3, .4, .5, .6, .6, .7, .8, .9] #training data set no. 1
        # y = [.6, .5, .4, .4, .1, .9, .4, .3, .5, .7]
        # label = [1, 1, 1, 1, 1, -1, -1, -1, -1, -1]

        # N = 4
        # x = [.2, .3, .7, .9] #training data set no. 2, showing XOR
        # y = [.2, .8, .2, .8]
        # label = [1, -1, -1, 1]

        N = 50
        x = []
        y = []
        label = []
        p = Sample_data_generator(N)
        training_data= p.X
        m_true = -training_data[0][2][1]/training_data[0][2][2], -training_data[0][2][0]/training_data[0][2][2] #get true values from the p object
        #print training_data
        #print training_data[0][2][0]
        for i in range(0,N):
			x.append(training_data[i][0][1])
			y.append(training_data[i][0][2])
			label.append(training_data[i][1])
        #print label

        #to find k_max
        x_ave = 0
        y_ave = 0
        for i in range(0,N):
        	x_ave = x_ave+x[i]
        	y_ave = y_ave+y[i]
        x_ave = x_ave/N
        y_ave = y_ave/N
        R0 = 0 #initialize
        for i in range(0,N):
        	R_guess = find_Radius(x[i], y[i], x_ave, y_ave)
        	if R_guess > R0:
        		R0 = R_guess
        margin0 = 99 #initialize
        for i in range(0,N):
        	margin_guess = find_margin(x[i], y[i], m_true)
        	if margin_guess < margin0:
        		margin0 = margin_guess
        k_max = R0*R0/(margin0*margin0)
       
        m = [1, 0.1] #m are my weigts, [0.01, 1], [-4, 8] #random guess
        learning_rate = 0.01 #random guess
        learning_rate_b = 0.01 #random guess
        max_attempts = 200 #20 #maximum attempts allowed to learn a given data point. Might be redundant. If max attempts are exceeded, increase learning rate.
        total_classification_errors = 1 #initialize
        epoch = 1 #initialize; to count how many times learing is done over given data set
        max_epochs = 10000000
        learning_iterations = 0 #to count how many iterations needed for convergence

        #plots_for_gif(N, learning_rate, x, y, m, m_true,label, 0, 0, 0, True) #uncomment to get first image for gif

        while total_classification_errors > 0 and epoch < max_epochs:
	        for i in range(0,N):
	            attempts = 1
	            classification = 1 #initialize with a bad classication

	            while classification == 1 and attempts < max_attempts:
	                classification = classify_point(x[i], y[i], m, label[i]) #classify point x[i], y[i]

	                if classification == 1: #if classification is bad, weights are shifted/trained.
	                	perceptron_output = -1*label[i] #technically it should be -1/label[i], but since labels are -1 or +1, it ends up being equivalent.
	                	m[0] = m[0] + learning_rate*(label[i]-perceptron_output)*x[i]
	                	m[1] = m[1] + learning_rate_b*(label[i]-perceptron_output)
	                	learning_iterations = learning_iterations + 1 #to count number of weight iterations needed for convergence
	                	#plots_for_gif(N, learning_rate, x, y, m, m_true, label, epoch, i, attempts, True) #uncomment to get images for gif
	                
	                #print i, attempts, classification
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

			print epoch, total_classification_errors, learning_iterations, k_max
			epoch = epoch + 1

	plots_for_gif(N, learning_rate, x, y, m, m_true, label, epoch, i, learning_iterations, True)
	print ('x_ave: %.3f, y_ave: %.3f, R0: %.3f, margin0: %.3f, k_max %.0f'%(x_ave, y_ave, R0, margin0, k_max))
	print ('True slope, Learnt slope') 
	print (m_true, m)
	print ('learning_iterations: %d'%(learning_iterations))

main()
#plt.show()
fig = plt.figure(2)

