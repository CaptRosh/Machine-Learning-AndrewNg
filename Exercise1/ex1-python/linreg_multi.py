import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits import mplot3d

def normalize(X):
    X = (X - X.describe().loc["mean"])/X.describe().loc["std"]
    return X

def normalizedCost(X,y):
    return np.dot(np.dot(np.linalg.inv(np.dot(X.T,X)),X.T),y)

def normalizeParam(X):
    X = (X - Xdf.describe().loc["mean"])/Xdf.describe().loc["std"]
    return X

def computeCost(X,y,theta):
    m = len(y)
    J = 0
    for i in range(m):
        J = J + ((np.dot(theta.T,X[i]) - y[i])**2)
    return J * 1/(2*m)

def gradientDescent(X,y,theta,alpha,iter):
    m = len(y)
    J_hist = np.zeros(iter+1)
    for _ in range(iter):
        for i in range(m):
            h_theta = np.dot(theta.T,X[i])
            temp0 = theta[0] - (alpha/m) * ((h_theta - y[i]))
            temp1 = theta[1] - (alpha/m) * ((h_theta - y[i])*X[i][1])
            temp2 = theta[2] - (alpha/m) * ((h_theta - y[i])*X[i][2])
            theta[0] = temp0
            theta[1] = temp1
            theta[2] = temp2
            J_hist[_] = computeCost(X,y,theta)
    return theta,J_hist
        
def predict(X,theta):
    return np.dot(X,theta)


df = pd.read_csv("ex1data2.txt",header=None)

Xdf = df[[0,1]]
ydf = df[[2]]

Xdf_norm = normalize(Xdf)

Xdf_norm.insert(0,None,1)

X_norm = Xdf_norm.to_numpy()
y= ydf.to_numpy()

theta = normalizedCost(X_norm,y)

X_test = normalizeParam(pd.DataFrame([[1650,3]]))
X_test.insert(0,None,1)

print(f"The price for a house with 1650 square feet area and 3 bedrooms via Normal Equations is : ${float(predict(X_test,theta)):.2f}\n")

alpha = 0.01
iterations = 400

init_theta = np.array([0,0,0])
print(f"Initial theta: {init_theta[0]},{init_theta[1]},{init_theta[2]}")
print(f"Cost function value: {float(computeCost(X_norm,y,init_theta)):.2f}\n")

init_theta = np.array([-2.14,1.45,2.08])
print(f"Test theta: {init_theta[0]},{init_theta[1]},{init_theta[2]}")
print(f"Cost function value: {float(computeCost(X_norm,y,init_theta)):.2f}\n")

theta_grad,J_hist = gradientDescent(X_norm,y,init_theta,alpha,iterations)
print(f"theta by gradient descent is: {float(theta_grad[0]):.2f},{float(theta_grad[1]):.2f},{float(theta_grad[2]):.2f}\n")

#Plot J(theta) curve vs. iterations
plt.plot(range(len(J_hist)),J_hist)
plt.ylabel("Cost function")
plt.xlabel("Iterations")
plt.draw()

print(f"The price for a house with 1650 square feet area and 3 bedrooms via Gradient Descent is : ${float(predict(X_test,theta_grad)):.2f}")

plt.show()