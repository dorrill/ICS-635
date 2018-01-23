# ICS-635

Please add description of Homework 2.

Anirvan's perceptron code:
Creates a random true line, and generates data points with labels about that line
OR
can take user-defined training set with labels and a true line
Then, it uses the perceptron model to try to learn the true line
N, total_classification_errors, learning_rate, number of iterations for convergence, R0, margin0, and k_max are appended into output.csv
The training data: (N, x, y, label, and true line) are dumped into training_data.txt

Variables to play with:
N, number of data points in sample; user-defined
m, my weigts; it can be random guess or user-defined
learning_rate, parameter m[0] or slope; user-defined
learning_rate_b, parameter m[1] or y-intercept; user-defined
max_epochs, number of runs allowed over data set to learn true line; user-defined
