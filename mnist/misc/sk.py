import numpy as np
import matplotlib.pyplot as plt
from sklearn.svm import SVC

# Given data
labels = np.array([-1, -1, -1, -1, 1, 1, 1, 1, 1, 1])
coordinates = np.array([[0, 0], [2, 0], [3, 0], [0, 2], [2, 2], [5, 1], [5, 2], [2, 4], [4, 4], [5, 5]])

# Create an SVM classifier
svm = SVC(kernel='linear')

# Fit the classifier to the data
svm.fit(coordinates, labels)

# Get the support vectors
support_vectors = svm.support_vectors_

# Get the weights and bias of the decision boundary
w = svm.coef_[0]
b = svm.intercept_[0]

# Create a meshgrid of points to plot the decision boundary
x_min, x_max = np.min(coordinates[:, 0]) - 1, np.max(coordinates[:, 0]) + 1
y_min, y_max = np.min(coordinates[:, 1]) - 1, np.max(coordinates[:, 1]) + 1
xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.02), np.arange(y_min, y_max, 0.02))
Z = svm.predict(np.c_[xx.ravel(), yy.ravel()])
Z = Z.reshape(xx.shape)

# Plot the data points
plt.scatter(coordinates[:, 0], coordinates[:, 1], c=labels, cmap='bwr', edgecolor='k')

# Plot the decision boundary
plt.contourf(xx, yy, Z, cmap='bwr', alpha=0.2)

# Plot the support vectors
plt.scatter(support_vectors[:, 0], support_vectors[:, 1], s=100, facecolors='none', edgecolors='k')

# Set plot limits and labels
plt.xlim(x_min, x_max)
plt.ylim(y_min, y_max)
plt.xlabel('x')
plt.ylabel('y')
plt.title('SVM with Support Vectors')

# Show the plot
plt.show()