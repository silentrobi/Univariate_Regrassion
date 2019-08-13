from matplotlib import pyplot as plt
import pandas as pd
import pylab as pl
import numpy as np
from sklearn import linear_model

#+++++++ READ CSV File ++++++++
#CSV file is in UTF-8 format. So, we explicitly mention encoding format
df = pd.read_csv("FuelConsumption.csv", encoding='utf-8')
# return dataset snapshot
print(type(df))
df.head()
# summarize the data
df.describe()

#++++++++ selecting features +++++++++
modelDf= df[['ENGINESIZE', 'CO2EMISSIONS']] # x= 'ENGINESIZE' and y= ENGINESIZE

# #+++++++ Histogram Plotting the feature and actual output+++++++++
visualizeData = modelDf
visualizeData.hist()
plt.show()
# #+++++++ Scatter Plotting the feature and actual output+++++++++
plt.scatter(x=visualizeData['ENGINESIZE'], y=visualizeData['CO2EMISSIONS'], color= 'red' )
plt.xlabel("ENGINESIZE")
plt.ylabel("CO2EMISSIONS")
plt.show()
