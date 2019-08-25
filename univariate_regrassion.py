from matplotlib import pyplot as plt
import pandas as pd
import pylab as pl
import numpy as np
from sklearn import linear_model

#+++++++ READ CSV File ++++++++
#CSV file is in UTF-8 format. So, we explicitly mention encoding format
df = pd.read_csv("FuelConsumption.csv", encoding='utf-8')
# return dataset snapshot
df.head()
# summarize the data
df.describe()

#++++++++ selecting features +++++++++
modelDf= df[['ENGINESIZE', 'CO2EMISSIONS']] # x= 'ENGINESIZE' and y= ENGINESIZE
print (modelDf[['ENGINESIZE']])
# #+++++++ Histogram Plotting the feature and actual output+++++++++
visualizeData = modelDf
visualizeData.hist()
plt.show()
# #+++++++ Scatter Plotting the feature and actual output+++++++++
plt.scatter(x=visualizeData['ENGINESIZE'], y=visualizeData['CO2EMISSIONS'], color= 'red' )
plt.xlabel("ENGINESIZE")
plt.ylabel("CO2EMISSIONS")
plt.show()
#+++++++++++ Creating train and test dataset +++++++++++
#split our dataset into train and test sets, 80% of the entire data for training, and the 20% for testing. 
# We create a mask to select random rows using np.random.rand() function

msk = np.random.rand(len(visualizeData)) < 0.8
train = modelDf[msk] #80 %
test = modelDf[~msk]
#+++++++++Train data visualization+++++++++++
plt.scatter(x=train['ENGINESIZE'], y=train['CO2EMISSIONS'], color= 'blue' )
plt.xlabel("ENGINESIZE")
plt.ylabel("CO2EMISSIONS")
plt.show()

#+++++++++++++Modeling data +++++++++++++++
from sklearn import linear_model
regression = linear_model.LinearRegression()
#numpy.asanyarray Convert the input to an ndarray, but pass ndarray subclasses through.
train_x = np.asanyarray(train[['ENGINESIZE']])
train_y = np.asanyarray(train[['CO2EMISSIONS']])
regression.fit (train_x, train_y)
# The coefficients
print ('Coefficients: ', regression.coef_)
print ('Intercept: ',regression.intercept_)

#+++++++++++Plot outputs+++++++++++++++

plt.scatter(train['ENGINESIZE'], train['CO2EMISSIONS'],  color='blue')
#regression.intercept_[0] is theta_0 and regression.coef_[0][0] is theta_1
plt.plot(train_x, regression.coef_[0][0]*train_x + regression.intercept_[0], '-r')
plt.xlabel("Engine size")
plt.ylabel("Emission")
plt.show()


#++++++++++ Evaluation ++++++++++++
#we compare the actual values and predicted values to calculate the accuracy of a regression model. Evaluation metrics provide a key role in the development of a model,
# as it provides insight to areas that require improvement.
from sklearn.metrics import r2_score

test_x = np.asanyarray(test[['ENGINESIZE']])
test_y = np.asanyarray(test[['CO2EMISSIONS']])
test_y_hat = regression.predict(test_x)

print("Mean absolute error: %.2f" % np.mean(np.absolute(test_y_hat - test_y)))
print("Residual sum of squares (MSE): %.2f" % np.mean((test_y_hat - test_y) ** 2))
#R-squared is not error, but is a popular metric for accuracy of your model.
# It represents how close the data are to the fitted regression line.
# The higher the R-squared, the better the model fits your data.
# Best possible score is 1.0 and it can be negative (because the model can be arbitrarily worse).
print("R2-score: %.2f" % r2_score(test_y_hat , test_y) )