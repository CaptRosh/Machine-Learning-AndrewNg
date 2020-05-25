import numpy as np
import matplotlib.pyplot as plt
import scipy.io as io
import scipy.optimize as op

def linRegCost(theta,X,y,lambda_val):
    
    row = X.shape[0]

    h_theta = np.dot(X,theta)
    
    if lambda_val != 0:
        return 1/(2*row) * np.dot((h_theta - y).T,(h_theta - y)) + (lambda_val/(2*row) * np.dot(theta[1:],theta[1:].T))
    else:
        return 1/(2*row) * np.dot((h_theta - y).T,(h_theta - y))

def linRegGrad(theta,X,y,lambda_val):
    row = X.shape[0]

    h_theta = np.dot(X,theta)
    
    theta_ = theta
    theta_[0] = 0

    return 1/row * np.dot((h_theta - y).T,X) + lambda_val/row * theta_.T


def trainLinReg(X,y,lambda_val):

    theta = np.zeros([X.shape[1],1])
    theta = op.fmin_cg(f=linRegCost,x0=theta.flatten(),args=(X,y.flatten(),lambda_val),fprime=linRegGrad,disp=False,maxiter=100)

    return theta

def learningCurve(X,y,Xval,yval,lambda_val):
    row = X.shape[0]
    
    error_train = np.zeros([row,1])
    error_val = np.zeros([row,1])

    for i in range(1,row):
        X_ = X[:i,:]
        y_ = y[:i]

        theta = trainLinReg(X_,y_,lambda_val).reshape([X_.shape[1],1])

        error_train[i] = linRegCost(theta,X_,y_,0)
        error_val[i] = linRegCost(theta,Xval,yval,0)

    return error_train[1:],error_val[1:]

def polyFeatures(X,p):
    X_poly = np.zeros([X.shape[0],p])

    for k in range(X.shape[0]):
        poly = np.zeros([p,1])
        
        for i in range(p):
            poly[i] = X[k]**(i+1)
        X_poly[k,:] = poly.reshape((8,))
        
    return X_poly

def featureNormalize(X):
    
    mu = np.mean(X,axis=0)
    X_norm = X - mu
    sigma = np.std(X_norm,axis=0,ddof=1)

    return ((X - mu)/sigma),mu, sigma

def elementNormalize(X,mu,sigma):
    return ((X-mu)/sigma)

def plotFit(X,mu,sigma,theta,p):
    x = np.arange(np.min(X)-15,np.max(X)+25,0.05)

    X_poly = polyFeatures(x,p)
    X_poly = elementNormalize(X_poly,mu,sigma)
    X_poly = np.append(np.ones([X_poly.shape[0],1]), X_poly, axis=1)

    plt.plot(x,np.dot(X_poly,theta),"--")

def validationCurve(X,y,Xval,yval):

    lambda_vec = np.array([0 ,0.001, 0.003, 0.01, 0.03, 0.1, 0.3, 1, 3, 10])

    error_train = np.zeros([lambda_vec.shape[0],1])
    error_val = np.zeros([lambda_vec.shape[0],1])

    for i in range(lambda_vec.shape[0]):
        theta = trainLinReg(X,y,lambda_vec[i]).reshape([X.shape[1],1])

        error_train[i] = linRegCost(theta,X,y,0)
        error_val[i] = linRegCost(theta,Xval,yval,0)

    return lambda_vec,error_train,error_val

print("Loading and Visualizing Data...\n")

data = io.loadmat("ex5data1.mat")

init_X = data["X"]
y = data["y"]
init_Xval = data["Xval"]
yval = data["yval"]
Xtest = data["Xtest"]
ytest = data["ytest"]

X = np.append(np.ones([init_X.shape[0],1]),init_X,axis=1)
Xval = np.append(np.ones([init_Xval.shape[0],1]),init_Xval,axis=1)

row = X.shape[0]

plt.plot(init_X,y,"rx")
plt.xlabel("Change in water level (x)")
plt.ylabel("Water flowing out of the dam (y)")
plt.legend(["Data"])
plt.draw()
plt.pause(1.5)


init_theta = np.array([[1],[1]])
init_lambda = 1

init_J = linRegCost(init_theta,X,y,init_lambda)
init_grad = linRegGrad(init_theta,X,y,init_lambda)

print(f"Cost at theta(1,1): {float(init_J):.6f}")
print(f"This value should be 303.993192\n")
print(f"""Gradient at theta(1,1): {float(init_grad[0][0]):.6f}, {float(init_grad[0][1]):.6f}""")
print(f"This value should be [-15.303016; 598.250744]\n")

trained_theta = trainLinReg(X,y,0)

plt.plot(init_X, np.dot(X,trained_theta),"--")
plt.legend(["Data","Linear Regression Line"])
plt.draw()
plt.pause(1.5)
plt.close()

err_lambda = 0

init_error_train,init_error_val = learningCurve(X,y,Xval,yval,err_lambda)

plt.plot(range(row-1),init_error_train,range(row-1),init_error_val)
plt.xlabel("Number of training samples")
plt.ylabel("Error")
plt.legend(["Training Error","Cross Validation Error"])
plt.draw()
plt.pause(1.5)
plt.close()

print("#Trn. Ex\tTrainError\tCross Validation Error")
for i in range(len(init_error_train)):
    print(f"{i}\t\t{float(init_error_train[i]):.6f}\t\t{float(init_error_val[i]):.6f}")
print("\n")

power = 8

X_poly = polyFeatures(init_X, power)
X_poly,mu,sigma = featureNormalize(X_poly)
X_poly = np.append(np.ones([X_poly.shape[0],1]),X_poly,axis=1)

X_poly_test = polyFeatures(Xtest, power)
X_poly_test = elementNormalize(X_poly_test, mu, sigma)
X_poly_test = np.append(np.ones([X_poly_test.shape[0],1]),X_poly_test,axis=1)

X_poly_val = polyFeatures(init_Xval, power)
X_poly_val = elementNormalize(X_poly_val, mu, sigma)
X_poly_val = np.append(np.ones([X_poly_val.shape[0],1]),X_poly_val,axis=1)

init_poly_lambda = 0

theta = trainLinReg(X_poly,y,init_poly_lambda)

plt.plot(init_X,y,'rx')
plotFit(init_X,mu,sigma,theta,power)
plt.xlabel('Change in water level (x)')
plt.ylabel('Water flowing out of the dam (y)')
plt.title(f'Polynomial Regression Fit (lambda = {init_poly_lambda:.2f})')
plt.draw()
plt.pause(1.5)
plt.close()

poly_error_train,poly_error_val = learningCurve(X_poly,y,X_poly_val, yval,err_lambda)


plt.plot(range(row-1),poly_error_train,range(row-1),poly_error_val)
plt.title(f'Polynomial Regression Learning Curve (lambda = {err_lambda:.2f})')
plt.xlabel('Number of training examples')
plt.ylabel('Error')
plt.axis([0,13,0,100])
plt.legend(['Train', 'Cross Validation'])
plt.draw()
plt.pause(1.5)
plt.close()

print("#Trn. Ex\tTrainError\tCross Validation Error")
for i in range(len(poly_error_train)):
    print(f"{i}\t\t{float(poly_error_train[i]):.6f}\t\t{float(poly_error_val[i]):.6f}")
print("\n")

lambda_vec,lambda_error_train,lambda_error_val = validationCurve(X_poly,y,X_poly_val,yval)

plt.plot(lambda_vec,lambda_error_train,lambda_vec,lambda_error_val)
plt.legend(['Train', 'Cross Validation'])
plt.xlabel('lambda')
plt.ylabel('Error')

print("Lambda Value\tTrainError\tCross Validation Error")
for i in range(len(lambda_error_train)):
    print(f"{lambda_vec[i]}\t\t{float(lambda_error_train[i]):.6f}\t\t{float(lambda_error_val[i]):.6f}")

plt.show()