from tkinter.tix import NoteBook
from pydataset import data
import pandas as pd 
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from matplotlib import pyplot as plt

pima = data('Pima.tr') #american women living in Phoenix

#pima.plot(kind='scatter', x='skin', y='bmi')
#plt.show()

#Test train split for supervised training
x_train,x_test,y_train,y_test = train_test_split(pima.skin, pima.bmi)

#Test train split visualization
#plt.scatter(x_train, y_train,label='Training',color='r',alpha=.7)
#plt.scatter(x_test,y_test,label='Testing Data',color='g', alpha=.7)
#plt.legend()
#plt.show()
#rojo utilizado para realizar el linear model, esto va a ser probado en lo verde

#Create linear model and train it
LR = LinearRegression()
LR.fit(x_train.values.reshape(-1,1), y_train.values)

#Use model to predict on TEST Data

prediction = LR.predict(x_test.values.reshape(-1,1))


#Plot prediction line against actual test data
#plt.plot(x_test, prediction, label= 'Linear Regression', color='b')
#plt.scatter(x_test, y_test, label='Actual Test Data', color='g', alpha=.7)
#plt.legend()
#plt.show()



#Prediction VMT of woman with skin fold 50
LR.predict(np.array([[50]]))[0]
score = LR.score(x_test.values.reshape(-1,1), y_test.values)
#how accurate it is
print(score)

