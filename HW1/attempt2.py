import sys
import os
import numpy as np
from numpy import random as rad
from numpy import array

import matplotlib.pyplot as plt


N = 100 #number of points
min = -100
max = 100

## -- Preparing Canvas
plt.clf()

fig = plt.figure()
ax = fig.add_subplot(111)

ax.annotate("Class A", (min*.8,max), xytext=(min*.8,max))
ax.annotate("Class B", (max,min*.8), xytext=(max*.8,min*.8))

plt.grid()
plt.xlim(min*1.2,max*1.2)
plt.ylim(min*1.2,max*1.2)

plt.draw()

#Let's keep this 2D for simplicity
def Point_Generator(N, min, max):
    Range = max - min
    x = rad.rand(N,1)*Range
    for i in range(0,N):
        x[i] += min
    return x

#print Point_Generator(20, -10, 5)
m = 4
intercept = 3
def Designator_1(xcor, ycor, N):
    Des_vec = np.empty([N,1])
    for i in range(0,N):
        if xcor[i] - m*ycor[i] > intercept:
            Des_vec[i] = 1
        else:
            Des_vec[i] = 0
    return Des_vec


def dotproduct(x,y,N):
    sum = 0
    for i in range(0,N):
        sum += x[i]*y[i]
    return sum

def Function(xcor, ycor, w_xcor, w_ycor, bias):
    if xcor*w_xcor + ycor*w_ycor + bias > intercept:
        return 1
    else:
        return 0

def graph_expected(formula, min, max):
    x = np.linspace(min, max, 200)
    y = eval(formula)
    plt.plot(x,y, color = "black",linestyle = 'dashed')

def graph_LEARNED(formula, min, max):
    x = np.linspace(min, max, 200)
    y = eval(formula)
    plt.plot(x,y, color = "purple",linestyle = 'solid')
    


def main():
        x = Point_Generator(N, min, max)
        y = Point_Generator(N, min, max)
        #print x,y
        w = np.empty([3,1])
        w[0] = 1
        w[1] = 0.1
        w[2] = 0.1
        rate = 0.0002

        Answers = Designator_1(x,y,N)
        max_attempt = 10000

        graph_expected("x/{} + {}".format(m, -1*intercept/m), min, max)
        

        

        for i in range(0,N):
            attempts = 0
            Guess = Function(x[i], y[i], w[1], w[2], w[0])
            Diff = Answers[i] - Guess
            while Diff != 0 and attempts < max_attempt:
                #print attempts
                w[0] += rate * float(Diff)
                w[1] += rate * float(Diff)*x[i]
                w[2] += rate * float(Diff)*y[i]
                #print w
                attempts += 1
            if attempts >= max_attempt:
                print("I give up with point",i)
            if Diff == 0:
                if Answers[i] == 1:
                    ax.plot(x[i], y[i], "c.",  marker ='x', ms=10)
                else:
                    ax.plot(x[i], y[i], "c.",  marker ='o', ms=10)
            else:
                ax.plot(x[i], y[i], "r.", ms=10)

        mag_w = np.sqrt((dotproduct(w,w,3)))

        print (w)

        graph_LEARNED("x*{} + {}".format(-w[1]/w[2] ,-w[0]/w[2]), min, max)



            

main()
plt.show()
