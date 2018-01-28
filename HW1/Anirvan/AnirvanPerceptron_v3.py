# Anirvan's perceptron
# Creates a random true line, and generates data points with labels about that line
# OR
# can take user-defined training set with labels and a true line
# Then, it uses the perceptron model to try to learn the true line

# Output files:
# N, total_classification_errors, learning_rate, learning_iterations, R0, margin0, k_max, mean_error are appended into output.csv
# The training data: N, x, y, label, and true line are dumped into training_data.txt
# A PNG file with the training data and final/learnt weights is created in $(pwd)/Plots.
# A PNG file for each change in weight can also be created. They can be combined into a GIF for learning animation.

# Variables to play with:
# N, number of data points in sample; user-defined
# m, my weigts; it can be random guess or user-defined
# learning_rate, parameter m[0] or slope; user-defined
# learning_rate_b, parameter m[1] or y-intercept; user-defined
# max_epochs, number of runs allowed over data set to learn true line; user-defined

import matplotlib
# Force matplotlib to not use any Xwindows backend.
matplotlib.use('Agg')

import sys
import os, subprocess
import random
import numpy as np
from numpy import array
import math
import matplotlib.pyplot as plt
import copy

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
    def __init__(self, *argv):
        if argv: # A line is specfied, create random separated data about the given line
            self.V = np.array([argv[0][1],argv[0][0], -1]) #m0 = -V1/V2, m1 = -V0/V2; set V2 = 1
        else: # No line specified, create random separated data about a random line 
            xA,yA,xB,yB = [random.uniform(0.25*max,0.75*max) for i in range(4)] #ensures true line is near the center of the canvas
            self.V = np.array([xB*yA-xA*yB, yB-yA, xA-xB])
    def generate_points(self, N):
        X = []
        for i in range(N):
            x1,x2 = [random.uniform(min, max) for i in range(2)]
            x = np.array([1,x1,x2])
            s = int(np.sign(self.V.T.dot(x))) #generate labels
            X.append((x, s, self.V))
        return X
    def epsilon_calculation(self, M, final_weights): #epsilon is the error in the trained line. It is the area between the trained line and true line. Generated using monte carlo
        testing_data = self.generate_points(M)
        x_test = []
        y_test = []
        label_test = []
        total_test_classification_errors = 0
        for i in range(0,M):
            x_test.append(testing_data[i][0][1])
            y_test.append(testing_data[i][0][2])
            label_test.append(testing_data[i][1])
            test_classification = classify_point(x_test[i], y_test[i], final_weights, label_test[i]) #classify new test points x[i], y[i]
            total_test_classification_errors = total_test_classification_errors + test_classification
        return total_test_classification_errors/float(M)

def main():
        N = 10
        x = [.1, .2, .3, .4, .5, .6, .6, .7, .8, .9] #training data set no. 1
        y = [.6, .5, .4, .4, .1, .9, .4, .3, .5, .7]
        label = [1, 1, 1, 1, 1, -1, -1, -1, -1, -1]
        m_true = [-1, 0.9]

        # N = 4
        # x = [.2, .3, .7, .9] #training data set no. 2, showing XOR
        # y = [.2, .8, .2, .8]
        # label = [1, -1, -1, 1]
        # m_true = []

        # N = 50
        # x = [0.134, 0.969, 0.331, 0.734, 0.415, 0.04, 0.599, 0.315, 0.676, 0.934, 0.948, 0.83, 0.695, 0.618, 0.788, 0.592, 0.517, 0.81, 0.842, 0.385, 0.611, 0.11, 0.123, 0.848, 0.267, 0.668, 0.237, 0.851, 0.505, 0.955, 0.501, 0.351, 0.361, 0.128, 0.368, 0.613, 0.056, 0.887, 0.349, 0.608, 0.324, 0.617, 0.584, 0.763, 0.722, 0.548, 0.918, 0.64, 0.297, 0.103]
        # y = [0.45, 0.635, 0.496, 0.932, 0.118, 0.104, 0.015, 0.412, 0.844, 0.788, 0.559, 0.224, 0.504, 0.194, 0.153, 0.796, 0.458, 0.909, 0.098, 0.676, 0.725, 0.499, 0.279, 0.402, 0.745, 0.694, 0.484, 0.093, 0.875, 0.591, 0.584, 0.697, 0.08, 0.664, 0.188, 0.394, 0.992, 0.946, 0.909, 0.434, 0.702, 0.785, 0.373, 0.895, 0.469, 0.534, 0.874, 0.226, 0.412, 0.132]
        # label = [-1, 1, -1, 1, -1, -1, 1, -1, -1, 1, 1, 1, 1, 1, 1, -1, -1, 1, 1, -1, -1, -1, -1, 1, -1, -1, -1, 1, -1, 1, -1, -1, -1, -1, -1, -1, -1, 1, -1, -1, -1, -1, -1, 1, 1, -1, 1, 1, -1, -1]
        # m_true = [9.199220, -5.474948]

        #Randomize the order of data
        sample_training_data = list(zip(x, y, label)) #pack the three lists into one list object
        random.shuffle(sample_training_data) #randomize the list object
        x, y, label = zip(*sample_training_data) #unpack the list object into three lists.
        #print (x)
        #print (y)


        #Generate a random data set for training
        # N = 50
        # x = []
        # y = []
        # label = []
        # p = Sample_data_generator()
        # training_data= p.generate_points(N)
        # m_true = -training_data[0][2][1]/training_data[0][2][2], -training_data[0][2][0]/training_data[0][2][2] #get true values from the p object
        # # print training_data
        # # print training_data[0][2][0]
        # for i in range(0,N):
            # x.append(training_data[i][0][1])
            # y.append(training_data[i][0][2])
            # label.append(training_data[i][1])
        # # print label
        # x = np.around(x, decimals=3) #to reduce size of array
        # y = np.around(y, decimals=3) #to reduce size of array

 
        #Define and initilaize variables
        #m = [1, 0.1] #m are my weigts, [0.01, 1], [-4, 8] #initialize
        m = [random.uniform(-max, max) for i in range(2)] #random guess
        learning_rate = 0.1 #user-defined
        learning_rate_b = 0.1 #user-defined
        max_attempts = 200 #20 #maximum attempts allowed to learn a given data point. Might be redundant. If max attempts are exceeded, increase learning rate.
        total_classification_errors = 1 #initialize
        epoch = 1 #initialize; to count how many times learing is done over given data set
        max_epochs = 1000000
        learning_iterations = 0 #to count how many iterations needed for convergence
        m_initial_guess = copy.deepcopy(m) #to record the initial guess


        #to find k_max, the theoretical maximum number of iterations needed for guaranteed convergence 
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


        #Implement percepton model
        while total_classification_errors > 0 and epoch < max_epochs:
            for i in range(0,N):
                attempts = 1
                classification = 1 #initialize with a bad classication
                while classification == 1 and attempts < max_attempts:
                    classification = classify_point(x[i], y[i], m, label[i]) #classify point x[i], y[i]
                    #this is where learning happens
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
            #now test the final weights on the sample, to check if they still classify all training points correctly.
	    #for each point in sample, check if it is still correctly classified.
	    #if any paoint's classification is bad, repeat the learning process over the entire training set
            total_classification_errors = 0
            for i in range(0,N):
                classification = classify_point(x[i], y[i], m, label[i])
                total_classification_errors = total_classification_errors + classification
            if epoch < 10 or epoch%10000 == 0:
                print (epoch, total_classification_errors, learning_iterations, np.rint(k_max))
            epoch = epoch + 1
        print (epoch, total_classification_errors, learning_iterations, np.rint(k_max))


        error = [] #Training error or test error
        try:
            p
        except NameError:
            var_exists = False
        else:
            var_exists = True
        if var_exists == False: #no pre-defined p object. Finding error in learning of training data set
            p = Sample_data_generator(m_true)
        for i in range(100):
            error.append(p.epsilon_calculation(10000, m))
            print(i)
        mean_error = np.mean(error)

        
        plots_for_gif(N, learning_rate, x, y, m, m_true, label, epoch, i, learning_iterations, True)
        print ('x_ave: %.3f, y_ave: %.3f, R0: %.3f, margin0: %.3f, k_max: %.0f, mean_error: %f'%(x_ave, y_ave, R0, margin0, k_max, mean_error))
        print ('True, Learnt, Guess:')
        print (m_true, m, m_initial_guess)
        print ('Number of learning_iterations: %d'%(learning_iterations))
        
        #apppend output into output.csv
        f = open('output.csv','a')
        f.write('\n' + '%d, %d, %s, %d, %.3f, %.10f, %.0f, %f'%(N, total_classification_errors, learning_rate, learning_iterations, R0, margin0, k_max, mean_error))
        f.close()

	#dump training data-set into training_data.txt
        f2 = open('training_data.txt','w')
        f2.write('N = %d'%(N))
        f2.write('\n' + '\n' + 'x = [')
        for i in range(0,N):
            f2.write('%s, '%(x[i]))
        f2.write(']' + '\n' + '\n' + 'y = [')
        for i in range(0,N):
            f2.write('%s, '%(y[i]))
        f2.write(']' + '\n' + '\n' + 'label = [')
        for i in range(0,N):
            f2.write('%s, '%(label[i]))
        f2.write(']' + '\n' + '\n' + 'True slope, True y-intercept:')
        f2.write('\n' + 'm_true = [%f, %f]'%(m_true[0], m_true[1]))
        f2.close()

main()
#plt.show()
#fig = plt.figure(2)