# Implementation-of-Linear-Regression-Using-Gradient-Descent

## AIM:
To write a program to predict the profit of a city using the linear regression model with gradient descent.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1.Import the required library and read the dataframe.

2.Write a function computeCost to generate the cost function.

3.Perform iterations og gradient steps with learning rate.

4.Plot the Cost function using Gradient Descent and generate the required graph.
## Program:
```
/*
Program to implement the linear regression using gradient descent.
Developed by: BALA MURUGAN P
RegisterNumber: 212222230017 
*/
```
```

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

data=pd.read_csv("/content/ex1.txt",header=None)

plt.scatter(data[0],data[1])
plt.xticks(np.arange(5,30,step=5))
plt.yticks(np.arange(-5,30,step=5))
plt.xlabel("Population of City (10,000s)")
plt.ylabel("Profit ($10,000)")
plt.title("Profit Prediction")

def computeCost(x,y,theta):
  m=len(y)
  h=x.dot(theta)
  square_err=(h-y)**2
  return 1/(2*m)*np.sum(square_err)
  
data_n=data.values
m=data_n[:,0].size
x=np.append(np.ones((m,1)),data_n[:,0].reshape(m,1),axis=1)
y=data_n[:,1].reshape(m,1)
theta=np.zeros((2,1))

computeCost(x,y,theta)
def gradientDescent(x,y,theta,alpha,num_iters):
  m=len(y)
  J_history=[]

  for i in range(num_iters):
    predictions = x.dot(theta)
    error = np.dot(x.transpose(),(predictions -y))
    descent=alpha*1/m*error
    theta-=descent
    J_history.append(computeCost(x,y,theta))

  return theta,J_history
  theta,J_history = gradientDescent(x,y,theta,0.01,1500)
print("h(x) *"+str(round(theta[0,0],2))+"+"+str(round(theta[1,0],2))+"x1")

plt.plot(J_history)
plt.xlabel("Iteration")
plt.ylabel("$J(\theta)$")
plt.title("Cost frunction using Gradient Descent")

plt.scatter(data[0],data[1])
x_value=[x for x in range(25)]
y_value=[y*theta[1]+theta[0]for y in x_value]
plt.plot(x_value,y_value,color="r")
plt.xticks(np.arange(5,30,step=5))
plt.yticks(np.arange(-5,30,step=5))
plt.xlabel("Population of City (10,000s)")
plt.ylabel("Profit ($10,000)")
plt.ylabel("Profit predictions")

def predict(x,theta):
  predictions=np.dot(theta.transpose(),x)
  return predictions[0]

predict1=predict(np.array([1,3.5]),theta)*10000
print("For population = 35,000, we predict a profit of $"+str(round(predict1,0)))

predict2=predict(np.array([1,7]),theta)*10000
print("For population = 70,000, we predict a profit of $"+str(round(predict2,0)))
```

## Output:
![image](https://github.com/Bala1511/Implementation-of-Linear-Regression-Using-Gradient-Descent/assets/118680410/60c5ffa8-1f61-4244-8936-3e4b6e656057)
![image](https://github.com/Bala1511/Implementation-of-Linear-Regression-Using-Gradient-Descent/assets/118680410/126f68ba-7aec-49bf-92d7-1ac26e4814bc)
![image](https://github.com/Bala1511/Implementation-of-Linear-Regression-Using-Gradient-Descent/assets/118680410/98d8be21-4c8f-48b9-a7cb-c1f093193369)
![image](https://github.com/Bala1511/Implementation-of-Linear-Regression-Using-Gradient-Descent/assets/118680410/3e86d2ad-163e-4cce-b863-a0a42b811630)
![image](https://github.com/Bala1511/Implementation-of-Linear-Regression-Using-Gradient-Descent/assets/118680410/dc9e7aa5-ec7f-4450-9538-1273743ca049)
![image](https://github.com/Bala1511/Implementation-of-Linear-Regression-Using-Gradient-Descent/assets/118680410/97e4a516-88d7-4817-95aa-137bb16bc8c3)


## Result:
Thus the program to implement the linear regression using gradient descent is written and verified using python programming.
