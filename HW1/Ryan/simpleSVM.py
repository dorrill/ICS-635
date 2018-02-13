#Created and maintained by Ryan Dorrill with the help of matplotlib, numpy, svn libraries
import numpy as np
import matplotlib.pyplot as plt
from sklearn import svm, datasets

#	#	#	#	#	#	#	#	#	#	#	#	#	#	#	#	#	#	#	#	#	#	#	#	#	#	#	#	#	#

def create_data(range_min,range_max,N,dataFileName):
	data_file = open(dataFileName,"w")
	rand_data = np.random.uniform(range_min,range_max,[N,2]) #generated uniformly random array of [x,y] with N sets of numbers ranging from range_min, to max
	for i in range(0,len(rand_data)):
		data_file.write("%s %s\n" %(rand_data[i,0],rand_data[i,1]))
	return rand_data

def generate_correct_output(t_data,m,b):	#takes training data of x's, y's -> x = t[0][i], y = t[1][i]
	trained_outputs = []
	for i in range(len(t_data)):
		if t_data[i,1] - ((m*t_data[i,0]) + b) > 0:
			trained_outputs.append(1)
		else:
			trained_outputs.append(0)
	t_outputs = np.array(trained_outputs)
	return t_outputs

def scatterXYs(training_data,m,b):
	x1=-10
	y1=(m*x1)+b
	x2=10
	y2=(m*x2)+b
	plt.figure()
	plt.xlabel('x')
	plt.ylabel('y')
	plt.title('Scatter of Data and True Line')
	plt.axis([-20, 20, -20, 20])
	plt.plot([x1, x2], [y1, y2], 'k-', color = 'r',label='True line')

	for i in range(len(training_data)):
		if training_data[i][1] - ((m*training_data[i][0]) + b)> 0:
			plt.scatter(training_data[i][0],training_data[i][1],c='r')
		else:
			plt.scatter(training_data[i][0],training_data[i][1],c='b')
	plt.legend(loc=1)
	plt.show()

def plotTrueLineDataAndLearnedLine(training_data,m,b,svm_m,svm_b):
	x1=-10
	y1=(m*x1)+b
	x2=10
	y2=(m*x2)+b
	plt.figure()
	plt.xlabel('x')
	plt.ylabel('y')
	plt.title('True vs. Learned Line via SVM')
	plt.axis([-20, 20, -20, 20])
	plt.plot([x1, x2], [y1, y2], 'k-', color = 'r',label='Correct line')

	y1_learned = (svm_b+(svm_m*x1))
	y2_learned = (svm_b+(svm_m*x2))
	plt.plot([x1, x2], [y1_learned, y2_learned], 'k-', color = 'green',label='Learned line')
	for i in range(len(training_data)):
		if training_data[i][1] - ((m*training_data[i][0]) + b)> 0:
			plt.scatter(training_data[i][0],training_data[i][1],c='r')
		else:
			plt.scatter(training_data[i][0],training_data[i][1],c='b')
	plt.legend(loc=1)
	#plt.savefig('./plots/num_pts%s_lrate%s_tsigma%s_epoch%s.png'%(data_length,learning_rate,t_sigma,epoch))
	plt.show()


def applySVMtoData(machineData,classifiers):
	clf = svm.SVC(kernel='linear', C = 1.0)
	clf.fit(machineData,classifiers)
	return clf



weights = []
dataFile = "./machinedata.txt"
slope = 3		#training_slope
intercept = -1
data_min = -10
data_max = 10
numpts = 16

new_data = create_data(data_min,data_max,numpts,dataFile)
print new_data
dataclassifiers = generate_correct_output(new_data,slope,intercept)
learnedfunction = applySVMtoData(new_data,dataclassifiers)
machineWeights = learnedfunction.coef_[0]
svm_slope = -machineWeights[0]/machineWeights[1]	#These two steps convert from svm 'weights' and alpha constant to easily plottable slope, intercept
svm_intercept = -learnedfunction.intercept_[0]/machineWeights[1]


#scatterXYs(new_data,slope,intercept)
plotTrueLineDataAndLearnedLine(new_data,slope,intercept,svm_slope,svm_intercept)





