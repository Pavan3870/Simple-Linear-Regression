# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Importing the dataset
df= pd.read_csv('income.data.csv')
print(df.head())

# Ploting the dataset by using scatter plot
'''
plt.scatter(df.income,df.happiness,color="red",marker='*')
plt.xlabel('income')
plt.ylabel('happiness')
plt.show()
'''

# Defining X and y
X = df['income'].values.reshape(-1,1)
y = df['happiness'].values.reshape(-1,1)
#print(X)

# Splitting the dataset into the Training set and Test set
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)

#importing the  model
from sklearn.linear_model import LinearRegression
lr=LinearRegression()

#fitting the model
lr.fit(X_train, y_train)

#Predict
y_pred=lr.predict(X_test)
print("Predicted Values :",y_pred)

#printing Values
print('Vaue of Intercept :',lr.intercept_)
print('Value of Coefficient :',lr.coef_)

#model evaluation
from sklearn import metrics
rmse = np.sqrt(metrics.mean_squared_error(y_test, y_pred))
R2 = metrics.r2_score(y_test, y_pred)
print("RMSE Score :",rmse)
print("R2_Score :", R2)
#comparision of acual values and predicted values

dt = pd.DataFrame({'Actual': y_test.flatten(), 'Predicted': y_pred.flatten()})
print(dt)

#plotting the predicted and actual values

df1 = dt.head(100)
#df1.plot(kind='bar',figsize=(10,4))
df1.plot(kind='line',figsize=(10,4))
plt.show()


#Plotting
plt.scatter(X_train,y_train)
#plt.plot(X_train,lr.predict(X_train),color="red")
plt.plot(X_test,lr.predict(X_test),color="red")
plt.xlabel("income")
plt.ylabel("Happiness")
plt.show()

