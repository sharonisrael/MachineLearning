import numpy as np
from tensorflow_core.python.keras.models import load_model

import classificaton_preprocess_data
import classificaton_process_test_data
from tensorflow.keras.models import Sequential
from tensorflow.keras.optimizers import Adam
from tensorflow.python.keras.layers import Dense

from sklearn.metrics import confusion_matrix
import itertools
import matplotlib.pyplot as plt

# emratamodel

model = Sequential([
    # The input_shape parameter expects a tuple of integers that matches the shape of the input data,
    # this data is one dimensional. so we specify (1,) as the shape.
    Dense(units=16, input_shape=(1,), activation='relu'),
    # Note, if you don’t explicitly set an activation function, then Keras will use the linear activation function.
    Dense(units=32, activation='relu'),
    # 2 is two possible outputs: either a patient experienced side effects, or not experience side effects.
    # This time, the activation function we’ll use is softmax.
    Dense(units=2, activation='softmax')
])

model.summary()

# Optimization function SGD or RMSProp or Adam
# loss function how the loss is calculated / min square error / min absolute error
# metrics is an array of metrics to judge performance of model
model.compile(optimizer=Adam(learning_rate=0.0001), loss='sparse_categorical_crossentropy', metrics=['accuracy'])

# Take data from preprocess
# take the labels 0, 1
# how many in batch - Too many in batch will need more computing power but will be more accurate
# an epoch is a single run through of all the data. single epoch is one single pass of all the data through the network,
# so we will run batches until complete epoch and then start next epoch batches
# batches in epoch = training set size / batch_size
# shuffle=True is default. since the data is being shuffled each epoch, it will be in a different order for each pass.
# verbose specifies how much output to the console we want to see during each epoch of training (values can be 0, 1, 2)
model.fit(x=classificaton_preprocess_data.scaled_train_samples, y=classificaton_preprocess_data.train_labels, batch_size=10, epochs=30,
          shuffle=True, verbose=2)

"""
    The loss will reduce and the accuracy will raise
    0s - loss: 0.2760 - acc: 0.9286
"""

# shuffle=True is default. since the data is being shuffled each epoch, it will be in a different order for each pass.
# shuffle is important because if no shuffle then it will take the last 10% which is the 2nd for loop only
model.fit(x=classificaton_preprocess_data.scaled_train_samples, y=classificaton_preprocess_data.train_labels, validation_split=0.1, batch_size=10,
          epochs=30, shuffle=True, verbose=2)

"""
    Total 2100 samples: Train on 1890 samples, validate on 210 samples (10%)
    val_loss and val_acc is on the validation set - see how it's generalizing on data it did not see.
    Epoch 20/20
    It's not over fitting because training and validation has same accuracy
    0s - loss: 0.2760 - acc: 0.9286 - val_loss: 0.1412 - val_acc: 1.0000
"""

# returns tuple of predict results probability
# Each element in the predictions list is itself a list of length 2. The sum of the two values in each list is 1.
predictions = model.predict(x=classificaton_process_test_data.scaled_test_samples, batch_size=10, verbose=0)

for prediction in predictions:
    print(prediction)

# show the most probable prediction
# It's a list of classes like 0, 1.
# We don't have the label classification so we cannot judge how accurate these predictions are.
rounded_predictions = model.predict_classes(x=classificaton_process_test_data.scaled_test_samples, batch_size=10, verbose=0)

for prediction in rounded_predictions:
    print(prediction)

# Comparing vs. real results
cm = confusion_matrix(y_true=classificaton_process_test_data.test_labels, y_pred=rounded_predictions)
print('Configuration matrix')
print(cm)
"""
The result is 420 test results divided into categories

[[193  17]
 [ 10 200]]

Y-Axis True labels (No/Yes) and X-Axis Predeiced lables (No/Yes)
1 2
3 4
1 = 193 No true & No predicted
2 = 200 No True & Yes predicted
"""

# save a model with all weights and architecture nodes, state optimizer to continue tuning
model_file_path = 'D:/My_Programming/Python/Data/medical_trial_model.h5'
model.save(model_file_path)
new_model = load_model(model_file_path)

new_model.summary()
new_model.get_weights()

"""
model.save_weights('my_model_weights.h5')
# architecture
model2 = Sequential([
    Dense(units=16, input_shape=(1,), activation='relu'),
    Dense(units=32, activation='relu'),
    Dense(units=2, activation='softmax')
])
# weights
model2.load_weights('my_model_weights.h5')
"""

