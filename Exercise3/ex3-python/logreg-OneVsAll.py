import numpy as np
import matplotlib.pyplot as plt
import scipy.io as io
import scipy.optimize as op

def sigmoid(z):
    return 1/(1 + np.exp(-z))

def lrCostFunc(theta,X,y,lambda_val,fmin):
    m = len(y)
    h_theta = sigmoid(np.dot(X,theta))
    J = -1/m * ((np.dot(y.T,np.log(h_theta)) + np.dot((1-y.T),np.log(1-h_theta)))) + (lambda_val/(2*m) * np.dot(theta[1:].T,theta[1:]))
    if not fmin:
        theta[0] = 0
    else:
        return J
    grad = 1/m * (np.dot((h_theta - y).T,X)) + (lambda_val/m * theta.T)
    return J,grad

def lrGradFunc(theta,X,y,lambda_val,fmin):
    """
    only taking in fmin because of args in scipy.optimize.fmin_cg
    """
    m = len(y)
    h_theta = sigmoid(np.dot(X,theta))
    grad = 1/m * (np.dot((h_theta - y).T,X)) + (lambda_val/m * theta.T)
    return grad

def oneVsAll(X,y,num_labels,lambda_val):
    m,n = X.shape
    all_theta = np.zeros([num_labels,n+1])
    X = np.append(np.ones([m,1]),X,axis=1)
    initial_theta = np.zeros([n+1,1])
    for i in range(num_labels):
        all_theta[i,:] = op.fmin_cg(f=lrCostFunc,fprime=lrGradFunc,x0=initial_theta.flatten(),args=(X,np.where(y==i,1,0).flatten(),lambda_val,True),maxiter=50,disp=False)
        print(f"Cost function performed for class : {i}")
    return all_theta

def predictOneVsAll(X,theta):
    m,n = X.shape
    p = np.zeros([m,10])
    X = np.append(np.ones([m,1]),X,axis=1)
    p = sigmoid(np.dot(X,theta.T))
    return np.argmax(p,axis=1).reshape(m,1)

df = io.loadmat("ex3data1.mat")
X = df["X"]
y = df["y"] - 1 #adjusting for octave index values

fig,axis = plt.subplots(10,10,figsize=(8,8))

for i in range(10):
    for j in range(10):
        axis[i,j].imshow(X[np.random.randint(0,5000),:].reshape(20,20,order="F"))
        axis[i,j].axis("off")

plt.draw()

num_labels = 10  #0-9
input_size = 400 #20x20pixels

#Testing Cost Function
print(f"Testing Regularized Cost Function\n")
theta_test = np.array([[-2],[-1],[1],[2]])
X_test = np.append(np.ones([5,1]),np.arange(1,16).reshape(5,3,order="F")/10,axis=1)
y_test = np.where(np.array([[1],[0],[1],[0],[1]])>= 0.5,1,0)
lambda_test = 3

J_test,grad_test = lrCostFunc(theta_test,X_test,y_test,lambda_test,False)

print(f"Cost: {float(J_test):.6f}")
print(f"Expected Cost: 2.534819\n")
print(f"""Gradients:
{float(grad_test[0][0]):.6f}
{float(grad_test[0][1]):.6f}
{float(grad_test[0][2]):.6f}
{float(grad_test[0][3]):.6f}
""")
print(f"""Expected Gradients:
0.146561
-0.548558
0.724722
1.398003
\n""")

lambda_actual = 0.1

print(f"Training One-Vs-All Logistic Regression:\n")
op_theta = oneVsAll(X,y,num_labels,lambda_actual)

print(f"\nCalculating Accuracy:")
pred = predictOneVsAll(X,op_theta)
accuracy = np.sum(np.where(pred == y,1,0))/len(y) * 100

print(f"Accuracy of the training set: {accuracy}")

plt.show()