import numpy as np

x = np.random.normal(loc=1.0, scale=10, size=[1000, 50])
mean = np.mean(x, axis=0)
std = np.std(x, axis=0)
x_norm = (x - mean) / std

sum = np.sum(x, axis=1)
list_row_more_10 = sum > 10

A = np.eye(3)
B = np.eye(3)
print(np.vstack((A, B)))