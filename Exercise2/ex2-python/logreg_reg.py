import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import scipy.optimize as op

def sigmoid(z):
    return 1/(1+np.exp(-z))

def featureMap(X1,X2):
    if X1.shape:
        m = X1.shape[0]
    else:
        m = 1
    out = np.ones([m,1])
    degree = 6
    for i in range(1,degree+1):
        for j in range(i+1):
            out = np.append(out,((X1**(i-j)) * X2**j).reshape(m,1),axis=1)
    return out

def costFuncReg(theta,X,y,lambda_val):
    h_theta = sigmoid(np.dot(X,theta))
    J = -1/len(y) * (np.dot((y.T),(np.log(h_theta))) + np.dot((1-y.T),(np.log(1-h_theta)))) + (lambda_val/(2*len(y)) * np.dot(theta[1:].T,theta[1:]))
    grad = 1/len(y) * (np.dot((h_theta - y).T,X)) + (lambda_val/len(y) * theta.T)
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

def plotDecisionBoundary(theta):
    u = np.linspace(-1, 1.5, 50);
    v = np.linspace(-1, 1.5, 50);

    z = np.zeros([len(u), len(v)]);

    for i in range(len(u)):
        for j in range(len(v)):
            z[i,j] = np.dot(featureMap(u[i], v[j]),theta)
    z = z.T

    plt.contour(u,v,z,[0])

df = pd.read_csv("ex2data2.txt",header=None)

Xdf = df[[0,1]]
ydf = df[[2]]

pos_vals = df.loc[df[2] == 1]
neg_vals = df.loc[df[2] == 0]

ax1 = pos_vals.plot(kind='scatter',x=0,y=1,s=50,color='b',marker="+",label="Admitted")
ax2 = neg_vals.plot(kind='scatter',x=0,y=1,s=50,color='y',ax=ax1,label="Not Admitted")
ax1.set_xlabel("Exam 1 score")
ax2.set_ylabel("Exam 2 score")

plt.draw()

Xdf.insert(0,None,1)

X = Xdf.to_numpy()
y = ydf.to_numpy()

X_map = featureMap(X[:,1],X[:,2])
row,col = X_map.shape

init_lambda_val = 1
init_theta = np.zeros([col,1])

init_cost,init_grad = costFuncReg(init_theta,X_map,y,init_lambda_val)

print(f"Cost at initial theta (zeros): {float(init_cost):.6f}")
print(f"Expected cost (approx): 0.693")
print(f"""Gradient at initial theta (zeros) - first five values only:
{init_grad[0][0]:.6f}
{init_grad[0][1]:.6f}
{init_grad[0][2]:.6f}
{init_grad[0][3]:.6f}
{init_grad[0][4]:.6f}
""")
print(f"""Gradient at initial theta (zeros) - first five values only:
0.0085
0.0188
0.0001
0.0503
0.0115
\n""")

test_lambda_val = 10
test_theta = np.ones([col,1])

test_cost,test_grad = costFuncReg(test_theta,X_map,y,test_lambda_val)

print(f"Cost at test theta (ones) (with lambda 10): {float(test_cost):.6f}")
print(f"Expected cost (approx): 3.16")
print(f"""Gradient at test theta (zeros) - first five values only:
{test_grad[0][0]:.6f}
{test_grad[0][1]:.6f}
{test_grad[0][2]:.6f}
{test_grad[0][3]:.6f}
{test_grad[0][4]:.6f}
""")
print(f"""Gradient at initial theta (zeros) - first five values only:
0.3460
0.1614
0.1948
0.2269
0.0922
\n""")

op_theta = op.fmin_tnc(func=costFuncReg,x0=init_theta.flatten(),args=(X_map,y.flatten(),init_lambda_val),disp=0)[0]

print(f"Train Accuracy: {accuracy(X_map,y,op_theta)*100:.2f}%")
print(f"Expected accuracy (with lambda = 1): 83.1 (approx)")

plotDecisionBoundary(op_theta)

plt.show()