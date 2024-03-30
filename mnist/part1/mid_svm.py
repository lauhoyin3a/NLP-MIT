from sklearn.svm import LinearSVC
import numpy as np
import matplotlib.pyplot as plt
x = np.array([[0, 0], [2, 0], [3, 0], [0, 2], [2, 2], [5, 1], [5, 2], [2, 4], [4, 4], [5, 5]])
y = np.array([-1, -1, -1, -1, -1, 1, 1, 1, 1, 1])
classifier = LinearSVC(random_state=0)
classifier.fit(x, y)
theta=classifier.coef_[0]
theta_0=classifier.intercept_[0]
print(theta_0)
print(theta)
slope = -theta[0] / theta[1]
intercept = -theta_0 / theta[1]

# Calculate the support vectors' distance to the decision boundary
#margin = 1 / np.sqrt(np.sum(classifier.coef_ ** 2))
margin = 1 / np.linalg.norm(theta)
print(margin)
def hinge_loss(x, y, theta, theta_0):
    loss = 0
    for i in range(len(x)):
        margin = y[i] * (np.dot(theta, x[i]) + theta_0)
        loss += max(0, 1 - margin)
    return loss
hinge_loss1=hinge_loss(x/2,y/2,theta,theta_0)
print(hinge_loss1)
# Define the x-axis range
x_vals = np.linspace(min([point[0] for point in x]), max([point[0] for point in x]), 100)

# Calculate the y-axis values for the decision boundary and the margin boundaries
decision_boundary = slope * x_vals + intercept
margin_boundary_upper = decision_boundary + margin
margin_boundary_lower = decision_boundary - margin

# Plot the data points, decision boundary, and margin boundaries
plt.scatter([point[0] for point in x], [point[1] for point in x], c=y)
plt.plot(x_vals, decision_boundary, color='r', label='Decision Boundary')
plt.plot(x_vals, margin_boundary_upper, '--', color='g', label='Margin Boundary')
plt.plot(x_vals, margin_boundary_lower, '--', color='g')
plt.xlabel('Feature 1')
plt.ylabel('Feature 2')
plt.legend()
plt.title('Linear SVM with Margin Boundary')
plt.show()