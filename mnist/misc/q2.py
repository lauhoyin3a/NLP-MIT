import numpy as np

y=np.array([-1,-1,-1,-1,-1,1,1,1,1,1])
x=np.array([[0,0],[2,0],[1,1],[0,2],[3,3],[4,1],[5,2],[1,4],[4,4],[5,5]])
mistake=np.array([1,65,11,31,72,30,0,21,4,15])
def feature_transform(X):
    transformed_features = np.zeros((X.shape[0], 3))
    transformed_features[:, 0] = X[:, 0]**2
    transformed_features[:, 1] = np.sqrt(2) * X[:, 0] * X[:, 1]
    transformed_features[:, 2] = X[:, 1]**2
    return transformed_features

X=feature_transform(x)
print(X)
transformed_X=y[:, np.newaxis] * X
#print(transformed_X)

theta_0=np.sum(y*mistake,axis=0)
#print("theta_0: ",theta_0)


result = np.multiply(mistake[:, np.newaxis], transformed_X)
theta=np.sum(result,axis=0)
print(theta)
print(X)
predictions=np.sign(X.dot(theta)+theta_0)
print(predictions)