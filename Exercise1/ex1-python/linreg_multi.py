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

def predict(X,theta):
    return np.dot(X,theta)

alpha = 0.01
iterations = 400

df = pd.read_csv("ex1data2.txt",header=None)

Xdf = df[[0,1]]
ydf = df[[2]]

Xdf_norm = normalize(Xdf,)

Xdf_norm.insert(0,None,1)

X_norm = Xdf_norm.to_numpy()
y= ydf.to_numpy()

theta = normalizedCost(X_norm,y)

X_test = normalizeParam(pd.DataFrame([[1650,3]]))
X_test.insert(0,None,1)

print(predict(X_test,theta))