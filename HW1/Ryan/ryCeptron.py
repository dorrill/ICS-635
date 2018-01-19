#import sys
import matplotlib.pyplot as plt
import matplotlib.lines as mlines
import matplotlib.transforms as mtransforms
from matplotlib.dates import HourLocator, MinuteLocator, DateFormatter
import datetime
import numpy as np
#Perceptron Notes
#Takes input that is linearly separable
#number of weights = number coordinates in x + 1 (w0 = intercept / constant)


def generate_output(data_point,weights):	#Our main perceptron decision function
	sum = 0	
	for i in range(len(data_point)):
		sum += data_point[i]*weights[i]

	if sum > 0:
		return 1
	else:
		return 0

#Our desired linear separation: y = 3x - 1  -->  
# If y - 3x + 1 > 0 ---> 1 (above our line)
# else ---> 0
m = 3		#training_slope
b = -1		#intercept
def generate_training_output(t_data):	#takes training data of x's, y's -> x = t[0][i], y = t[1][i]
	trained_outputs = []
	for i in range(len(t_data[0])):
		if t_data[1][i] - ((m*t_data[0][i]) + b)> 0:
			trained_outputs.append(1)
		else:
			trained_outputs.append(0)
	return trained_outputs

def plot_stuff(trained_weights,training_data,epoch):
	x1=-10
	y1=(m*x1)+b
	x2=10
	y2=(m*x2)+b
	plt.figure()
	plt.xlabel('x')
	plt.ylabel('y')
	plt.title('High Learning Rate: %s pts, rate: %s, sigma=%s'%(data_length,learning_rate,t_sigma))
	plt.axis([-20, 20, -20, 20])
	plt.plot([x1, x2], [y1, y2], 'k-', color = 'r',label='Correct line')

	m_l = -1*trained_weights[1]/(trained_weights[2])
	b_l = -1*trained_weights[0]/(trained_weights[2])
	y1_learned = (b_l+(m_l*x1))
	y2_learned = (b_l+(m_l*x2))
	plt.plot([x1, x2], [y1_learned, y2_learned], 'k-', color = 'green',label='Learned line')
	for i in range(len(training_data[0])):
		if training_data[1][i] - ((m*training_data[0][i]) + b)> 0:
			plt.scatter(training_data[0][i],training_data[1][i],c='r')
		else:
			plt.scatter(training_data[0][i],training_data[1][i],c='b')
	plt.legend(loc=1)
	#plt.savefig('./plots/num_pts%s_lrate%s_tsigma%s_epoch%s.png'%(data_length,learning_rate,t_sigma,epoch))
	plt.show()



def update_weights(data,trained_outputs,weights,l_rate):	#takes training data and returns training outputs	
	dat_point = [1,0,0] #x0 = 1 by default because w0 = linear intercept
	for i in range(0,len(data[0])):
		dat_point[1] = data[0][i]	#take ith x value
		dat_point[2] = data[1][i]	#take ith y value
		output = generate_output(dat_point,weights)
		#print "output: ",output,' ',trained_outputs[i],"    diff: ",trained_outputs[i]-output
		if output != trained_outputs[i]:
			for j in range(len(weights)):
				#print "weight",j, " = ",weights[j], " + ",l_rate,"*(",trained_outputs[i],' - ',-1*output,")*",dat_point[j]
				weights[j] = weights[j] + (l_rate)*(trained_outputs[i]-output)*dat_point[j]
	return weights

data_length = 1000	#how many data points to create
learning_rate = 0.1

t_mu, t_sigma = -1, 4 # mean and standard deviation
training_x = np.random.normal(t_mu, t_sigma, data_length)
training_y = np.random.normal(t_mu, t_sigma, data_length)
training_data = (training_x,training_y)

training_outputs = generate_training_output(training_data)

rand_weights = (np.random.normal(0, 0.05,3)) #generate three training rates -> w_intercept,w_x,w_y
epochs = 9
trained_weights = update_weights(training_data,training_outputs,rand_weights,learning_rate)
plot_stuff(trained_weights,training_data,0)
for k in range(1,epochs):
	trained_weights = update_weights(training_data,training_outputs,trained_weights,learning_rate)
	plot_stuff(trained_weights,training_data,k)
	print "Trained weights: ",trained_weights/(trained_weights[2])





	
