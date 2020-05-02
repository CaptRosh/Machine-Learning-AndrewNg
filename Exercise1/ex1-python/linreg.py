import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits import mplot3d


alpha = 0.01
iterations = 1500
init_theta = np.zeros([2,1])

def computeCost(X,y,theta):
    m = len(y)
    J = 0
    for i in range(m):
        J = J + ((np.dot(theta.T,X[i]) - y[i])**2)
    return J * 1/(2*m)

def gradientDescent(X,y,theta,alpha,num_iter):
    m = len(y)

    for _ in range(num_iter):
        for i in range(m):
            h_theta = np.dot(theta.T,X[i])
            temp0 = theta[0] - (alpha/m * (h_theta - y[i]))
            temp1 = theta[1] - (alpha/m * (h_theta - y[i]) * X[i][1])
            theta[0] = temp0
            theta[1] = temp1
    return theta

def prediction(X,theta):
    return np.dot(theta.T,X)*10000

df = pd.read_csv("ex1data1.txt", header=None)

Xdf = df[[0]]
ydf = df[[1]]

#Plot the data
plt.plot(Xdf,ydf,'rx')

Xdf.insert(0,None,1)


X = Xdf.to_numpy()
y = ydf.to_numpy()

plt.draw()

print("""Testing the cost function ...
With theta = [0 ; 0]
Expected cost value (approx) 32.07""")

J = computeCost(X,y,init_theta)
print(f"Calculated Cost value {float(J):.2f}\n")

test_theta = np.array([-1,2])
print("""With theta = [-1 ; 2]
Expected cost value (approx) 54.24""")
print(f"Calculated Cost value {float(computeCost(X,y,test_theta)):.2f}\n")


print("""Expected theta values (approx) -3.6303,1.1664""")

theta = gradientDescent(X,y,init_theta,alpha,iterations)

#Plot linear regression line
plt.plot(X[:,1],np.dot(X,theta))
plt.draw()

print(f"Calculated Theta Values: {float(theta[0]):.2f},{float(theta[1]):.2f}\n")

print(f"For a population of 35000, we predict profit of {float(prediction(np.array([1,3.5]),theta)):.2f}")
print(f"For a population of 70000, we predict profit of {float(prediction(np.array([1,7]),theta)):.2f}")


theta0_vals = np.linspace(-10, 10, 100);
theta1_vals = np.linspace(-1, 4, 100);
J_vals = np.zeros((len(theta0_vals), len(theta1_vals)));


for i in range(len(theta0_vals)):
    for j in range(len(theta1_vals)):
        t = np.array([theta0_vals[i],theta1_vals[j]])
        J_vals[i,j] = computeCost(X,y,t)

#Plot the surface plot of J(theta) vs. values of theta
plt.figure()
ax = plt.axes(projection='3d',  proj_type='ortho')
ax.plot_surface(theta0_vals,theta1_vals,J_vals.T)
plt.draw()

#Plot a contour figure
plt.figure()
plt.contour(theta0_vals,theta1_vals,J_vals)
plt.axis([-50,50,-5,20])
plt.draw()
plt.show()