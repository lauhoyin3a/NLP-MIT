import numpy as np

# Given data
labels = np.array([-1, -1, -1, -1, -1, 1, 1, 1, 1, 1])
coordinates = np.array([[0, 0], [2, 0], [3, 0], [0, 2], [2, 2], [5, 1], [5, 2], [2, 4], [4, 4], [5, 5]])
mistakes = np.array([1, 9, 10, 5, 9, 11, 0, 3, 1, 1])

# Initialize theta and theta_0
theta = np.zeros(coordinates.shape[1])
theta_0 = 0

# Linear Perceptron Algorithm
for i in range(len(coordinates)):
    x = coordinates[i]
    y = labels[i]
    num_mistakes = mistakes[i]

    for _ in range(num_mistakes):
        if y * (np.dot(theta, x) + theta_0) <= 0:
            theta += y * x
            theta_0 += y

# Print the results
print("Theta:", theta)
print("Theta_0:", theta_0)