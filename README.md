# Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored

## AIM:
To write a program to predict the marks scored by a student using the simple linear regression model.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1.Import the standard Libraries.

2.Set variables for assigning dataset values.

3.Import linear regression from sklearn.

4.Assign the points for representing in the graph

5.Predict the regression for marks by using the representation of the graph.

6.Compare the graphs and hence we obtained the linear regression for the given datas.

## Program:
```
/*
Program to implement the simple linear regression model for predicting the marks scored.
Developed by: G MALARVIZHI
RegisterNumber: 212222040096

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
df=pd.read_csv('student.csv')
df.head(10)
plt.scatter(df['X'],df['Y'])
plt.xlabel('X')
plt.ylabel('Y')
x=df.iloc[:,0:-1]
y=df.iloc[:,-1]
from sklearn.model_selection import train_test_split
X_train,X_test,Y_train,Y_test=train_test_split(x,y,test_size=0.2,random_state=0)
from sklearn.linear_model import LinearRegression
lr=LinearRegression()
lr.fit(X_train,Y_train)
X_train
Y_train
lr.predict(x_test.iloc[0].values.reshape(1,1))
plt.scatter(df['X'],df['Y'])
plt.xlabel('X')
plt.ylabel('Y')
plt.plot(X_train,lr.predict(X_train),color='orange')
lr.coef_
lr.intercept_
```

## Output:

# 1)HEAD:

![image](https://github.com/22008650/Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored/assets/122548204/23e2ae8d-92ec-458c-912d-1c9a3ad2e86c)

# 2)GRAPH OF PLOTTED DATA:

![image](https://github.com/22008650/Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored/assets/122548204/ee05f108-4294-44db-a552-4c4f433c1f56)

# 3)TRAINED DATA:

![image](https://github.com/22008650/Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored/assets/122548204/c5bd439a-3bb9-40e0-ad56-14f1161a8f4a)

# 4)LINE OF REGRESSION:

![image](https://github.com/22008650/Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored/assets/122548204/1ba98bc8-9b54-4c8d-9274-f311a20e1791)

# 5)COEFFICIENT AND INTERCEPT VALUES:

![image](https://github.com/22008650/Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored/assets/122548204/a498a799-fbfd-4364-86ae-adcb1437b36b)

## Result:
Thus the program to implement the simple linear regression model for predicting the marks scored is written and verified using python programming.
