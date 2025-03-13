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

4.Assign the points for representing in the graph.

5.Predict the regression for marks by using the representation of the graph.

6.Compare the graphs and hence we obtained the linear regression for the given datas.

## Program:
/*
Program to implement the simple linear regression model for predicting the marks scored.
Developed by: RAJAMANIKANDAN R 
RegisterNumber: 212223220082
*/
```
import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error,mean_squared_error
import matplotlib.pyplot as plt

dataset=pd.read_csv('student_scores.csv')
print(dataset.head())
print(dataset.tail())
```
## Output:
![image](https://github.com/user-attachments/assets/3052d07d-089c-4b1e-b3af-669a9932cd6f)

##PROGRAM:
```
dataset.info()
```
##OUTPUT:
![image](https://github.com/user-attachments/assets/5cfef523-6623-4c66-beb2-435a052632ce)

##PROGRAM:
```
X=dataset.iloc[:,:-1].values
print(X)
Y=dataset.iloc[:,-1].values
print(Y)
```
##OUTPUT:
![image](https://github.com/user-attachments/assets/dad7c703-4ebe-43b0-991a-8f7f00f3814b)

##PROGRAM:
```
from sklearn.model_selection import train_test_split
X_train,X_test,Y_train,Y_test=train_test_split(X,Y,test_size=0.2,random_state=0)
X_train.shape
```
##OUTPUT:
![image](https://github.com/user-attachments/assets/9b397afc-487d-451d-b296-46f68508f573)

##PROGRAM:
```
X_test.shape
```
##OUTPUT:
![image](https://github.com/user-attachments/assets/283a191f-1009-4c81-8377-d7b82819eead)

##PROGRAM:
```
from sklearn.linear_model import LinearRegression
reg=LinearRegression()
reg.fit(X_train,Y_train)
```
##OUTPUT:
![image](https://github.com/user-attachments/assets/da49a5ab-38f0-40c8-9b6f-f51fd5f5273d)

##PROGRAM:
```
Y_pred=reg.predict(X_test)
print(Y_pred)
print(Y_test)
```

##OUTPUT:
![image](https://github.com/user-attachments/assets/40c8de35-2b3a-455e-83e7-e2cb0fe38b77)

##PROGRAM:
```
plt.scatter(X_train,Y_train,color='green')
plt.plot(X_train,reg.predict(X_train),color='yellow')
plt.title('Training Set(H vs S)')
plt.xlabel('Hours')
plt.ylabel('Scores')
plt.show()
```

#OUTPUT:
![image](https://github.com/user-attachments/assets/967bc443-d7f5-4e5f-aab7-2e9778f15005)

##PROGRAM:
```
plt.scatter(X_test,Y_test,color='red')
plt.plot(X_train,reg.predict(X_train),color='silver')
plt.title('Test Set(H vs S)')
plt.xlabel('Hours')
plt.ylabel('Scores')
plt.show()
```

##OUTPUT:
![image](https://github.com/user-attachments/assets/846fc938-7b12-44cd-93b4-d9648375465f)


## Result:
Thus the program to implement the simple linear regression model for predicting the marks scored is written and verified using python programming.
