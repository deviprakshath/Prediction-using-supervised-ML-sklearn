#here all the libraries needed for the given project
#Fast and versatile, the NumPy vectorization, indexing, and broadcasting concepts are the standards of array computing today.
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn  import metrics
from sklearn.metrics import r2_score

import time

'''acess the file using pandas
Here importing and reading of document passed which is given by the internship site to 
by copying the dat from site to notepad and as a csv file '''

data_set = pd.read_csv("D:\GRIPspark internshippp (python )\data_set1.xlsx")
print("\t\t\t\nThe Data succefully filled into the computer")

#This head() used to return the first 5 values form the data_set
print(data_set.head())

#data_set.shape command is used to return the number of column and rows in the file data
print(data_set.shape)

#The below command is used to describet the data values tp mean count, std, min ,25% ,50%
print(data_set.describe())
plt.scatter(data_set['Hours'], data_set['Scores'])
plt.title('hours vs % ')
plt.xlabel('total study Hours')
plt.ylabel('Score')
plt.show()
time.sleep(1)
plt.close()
#importing of required module into the project

x = data_set.iloc[:,:-1].values
y = data_set.iloc[:,-1].values

X_train , X_test,Y_train,Y_test = train_test_split(x,y,test_size=0.3,random_state=0)

regres = LinearRegression()
regres.fit(X_train,Y_train)


#plotting of regression line
line = regres.coef_*x+regres.intercept_

#the dat_set is plotted
plt.scatter(x,y)
plt.plot(x,line, color='blue')
plt.show()


#prediction of the data_set
y_pred= regres.predict(X_test)
print((y_pred))

#traing data set visualaization
plt.scatter(X_train, Y_train, color='yellow')
plt.plot(X_train,regres.predict(X_train), color = "blue")
plt.title('Hours vs %(training set) ')
plt.xlabel('Total study Hours')
plt.ylabel('percentage of marks')
plt.show()

# comparing the actual values with that of predicted ones
data_set = pd.DataFrame({'Actual_data_set':Y_test,'Predicted_data_set':y_pred})
print(data_set)

xvalue = int(input("enter the value"))
data_set = np.array(xvalue)
data_set = data_set.reshape(-1,1)
pred = regres.predict(data_set)
print("if the student studied for ",xvalue,"hours/day, the the score is {}.".format(pred))


#ERROR matrics
#for mean absolute errors
print('Mean absolute error;',metrics.mean_absolute_error(Y_test,y_pred))

# for R-Square of the model
print("the R-Square of the model is :",r2_score(Y_test,y_pred))

#CONCLUSION:
#Here i used linear Regression model to predict the score of the student taking the input as integer andd predicted score