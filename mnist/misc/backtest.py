import numpy as np

# Given data
labels = np.array([-1, -1, -1, -1, 1, 1, 1, 1])
coordinates = np.array([[0, 0], [2, 0], [0, 2], [2, 2], [5, 1], [5, 2], [2, 4], [4, 4]])
mistakes = np.array([1,6,1,5,3,1,2,0])

# Finding the support vectors
support_vectors = coordinates[mistakes > 0]
support_labels = labels[mistakes > 0]

w = np.dot(support_labels * support_vectors.T, support_labels)
b = np.mean(support_labels - np.dot(w, support_vectors.T))

# Calculate the sum of hinge losses for the maximum margin separator
hinge_losses = np.maximum(0, 1 - labels * (np.dot(w, coordinates.T) + b))
sum_hinge_losses_max_margin = np.sum(hinge_losses)

# Calculate the sum of hinge losses for the modified separator with halved parameters
w_modified = w / 2
b_modified = b / 2
hinge_losses_modified = np.maximum(0, 1 - labels * (np.dot(w_modified, coordinates.T) + b_modified))
sum_hinge_losses_modified = np.sum(hinge_losses_modified)

# Printing the results
print("Parameters for Maximum Margin Separator:")
print("w =", w)
print("b =", b)
print("Sum of hinge losses for Maximum Margin Separator:", sum_hinge_losses_max_margin)

print("\nParameters for Modified Separator (with halved parameters):")
print("w_modified =", w_modified)
print("b_modified =", b_modified)
print("Sum of hinge losses for Modified Separator:", sum_hinge_losses_modified)