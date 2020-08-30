from tensorflow import keras
from tensorflow.keras.datasets import mnist

# input image dimensions
img_rows, img_cols = 28, 28

# the data, split between train and test sets
# The MNIST data is split between 60,000 28 x 28 pixel training images and 10,000 28 x 28 pixel images
(x_train, y_train), (x_test, y_test) = mnist.load_data()

import matplotlib.pyplot as plt
plt.imshow(x_train[0]) #training image
plt.show()
print(y_train[0]) #what the number is supposed to be
