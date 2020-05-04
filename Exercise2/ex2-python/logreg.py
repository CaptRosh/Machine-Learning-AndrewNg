import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import scipy.optimize as op

def sigmoid(z):
    return 1/(1+np.exp(-z))


def costFunction(theta,X,y):
    m = len(y)
    h_theta = sigmoid(np.dot(X,theta))
    J = -1/m * (np.dot((y.T),(np.log(h_theta))) + np.dot((1-y.T),(np.log(1-h_theta))))
    grad = alpha/m * np.dot((h_theta - y).T,X)
    return J,grad

def predict(X,theta):   
    return sigmoid(np.dot(theta,X))

def accuracy(X,y,theta):
    prediction = predict(theta,X)
    true = 0
    for i in range(len(y)):
        if prediction[i] >= 0.5:
            prediction[i] = 1
        else:
            prediction[i] = 0
        if prediction[i] == y[i]:
            true += 1
    return true/len(y)

def plotDecisionBoundary(X,theta):
    plot_x = np.array([np.min(X[:,2])-2, np.max(X[:,2])+2])
    plot_y = -1/theta[2] * (theta[1]*plot_x + theta[0])
    plt.plot(plot_x,plot_y)

df = pd.read_csv("ex2data1.txt",header=None)

Xdf = df[[0,1]]

pos_vals = df.loc[df[2]==0]
neg_vals = df.loc[df[2]==1]
ax1 = pos_vals.plot(kind='scatter',x=0,y=1,s=50,color='b',marker="+",label="Admitted")
ax2 = neg_vals.plot(kind='scatter',x=0,y=1,s=50,color='y',ax=ax1,label="Not Admitted")
ax1.set_xlabel("Exam 1 score")
ax2.set_ylabel("Exam 2 score")

plt.draw()

Xdf.insert(0,None,1)
ydf = df[[2]]

X = Xdf.to_numpy()
y = ydf.to_numpy()


init_theta = np.array([[0],[0],[0]])
alpha = 0.03
iterations = 1000

J,grad = costFunction(init_theta,X,y)

print(f"Cost at initial theta[zeros]: {float(J):.4f}")
print(f"Expected Cost(approx): .693")
print(f"Gradient at initial theta(zeros):\n {float(grad[0][0]):.5f}\n {float(grad[0][1]):.5f}\n {float(grad[0][2]):.5f}")
print(f"Expected gradients (approx):\n -0.1000\n -12.0092\n -11.2628\n")

test_theta = np.array([[-24], [0.2], [0.2]])
test_J,test_grad = costFunction(test_theta,X,y)

print(f"Cost at test theta[-24,0.2,0.2]: {float(test_J):.4f}")
print(f"Expected Cost(approx): 0.218")
print(f"Gradient at test theta[-24,0.2,0.2]:\n {float(test_grad[0][0]):.5f}\n {float(test_grad[0][1]):.5f}\n {float(test_grad[0][2]):.5f}")
print(f"Expected gradients (approx):\n -25.161\n 0.206\n 0.201\n")

#Use only 1D arrays
op_theta = op.fmin_tnc(func = costFunction,x0 = test_theta.flatten(),args=(X,y.flatten()),disp=0)[0]

print(f"Theta after fmin_tnc:\n {float(op_theta[0]):.4f}\n {float(op_theta[1]):.4f}\n {float(op_theta[2]):.4}\n")

print(f"For a student with scores 45 and 85, we predict an admission probability of: {predict(np.array([1,45,85]),op_theta):.6f}")
print(f"Expected value: 0.775 +/- 0.002\n")

print(f"Model Accuracy: {accuracy(X,y,test_theta)*100}%")
print(f"Expected Accuracy: 89%")

plotDecisionBoundary(X,op_theta)
plt.show()