from functools import reduce

import pandas as pd #Data-handling Library

from tensorflow.keras.models import Sequential
from tensorflow.python.keras.layers import Dense
from tensorflow.keras.optimizers import Adam
import matplotlib.pyplot as plt

def Average(lst):
    return reduce(lambda a, b: a + b, lst) / len(lst)

df=pd.read_csv('D:/My_Programming/Python/Data/IowaHousingPrices.csv')
# len of 1460
squareFeet = df[['SquareFeet']].values #x value
# len of 1460
salePrice = df[['SalePrice']].values #y value

model = Sequential([
    # The input_shape parameter expects a tuple of integers that matches the shape of the input data,
    # this data is one dimensional. so we specify (1,) as the shape.
    # Activation -If you don't activation, no activation is applied (ie. "linear" activation: a(x) = x).

    # Feeding in one value which is the list/vector
    Dense(units=1, input_shape=(1,))
])

model.summary()


# Optimization function SGD or RMSProp or Adam
# Adam is fastest and most accurate optimizer
# learning rate is how fast is going to optimize. Small learning rate will take longer to get optimum
# it's the bow. lr=1 is fast but inaccurate
# loss function how the loss is calculated / min square error / min absolute error
# metrics is an array of metrics to judge performance of model
model.compile(optimizer=Adam(learning_rate=1), loss='mean_squared_error')

# Training samples is squareFeet
# Training labels is salePrice
# epoch is a single run through of all the data
model.fit(squareFeet,salePrice, epochs=30, batch_size=10)

df.plot(kind='scatter',
       x='SquareFeet',
       y='SalePrice', title='Housing Prices and Square Footage of Iowa Homes')

y_pred = model.predict(squareFeet) #The predicted housing price based on square feet

#print(len(y_pred[:,0]))
#print(y_pred[:,0])
#print(Average(y_pred[:,0]))

print(model.predict([2000]))
# [[235519.95]]

# [1208.1405  891.7354 1261.8164 ... 1653.0851  761.7833  887.4978]
# 2142.551198630137
#Plot the linear regression line
plt.plot(squareFeet, y_pred, color='red')
# show
plt.show()