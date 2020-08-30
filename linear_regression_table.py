import numpy as np
import pandas as pd #Data-handling Library
from tensorflow.keras.models import Sequential
from tensorflow.python.keras.layers import Dense
from tensorflow.keras.optimizers import Adam

df=pd.read_csv('D:/My_Programming/Python/Data/HousingPrices_table.csv')
house_data = df.drop(columns=['SalePrice'])
# print(house_data)
house_price = df[['SalePrice']].values #y value

model = Sequential([
    # input layer of 8 because of the 8 columns
    Dense(units=8, input_shape=(8,), activation='relu'),
    # no input layer because internal
    Dense(units=8, activation='relu'),
    # output 1 single value of sales price
    Dense(units=1)
])

model.summary()

model.compile(optimizer=Adam(), loss='mean_squared_error')

model.fit(house_data,house_price, epochs=30, batch_size=10)

test_data = np.array([2003,	854, 1710, 2, 1, 3, 8, 2008])
print(test_data.reshape(1,8))
print(model.predict(test_data.reshape(1,8), batch_size=1))




