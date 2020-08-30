import numpy as np
from sklearn.metrics import confusion_matrix
from tensorflow import keras
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.python.keras import Sequential
from tensorflow.python.keras.layers import Dense
from tensorflow.keras.optimizers import Adam

model_folder = 'D:/My_Programming/Python/Data/cats-and-dogs'
train_path = model_folder + '/train/'
valid_path = model_folder + '/valid/'
test_path = model_folder + '/test/'

"""
Keras assumes that images are stored in a folder tree with one separate subfolder per class, like this:

    some/path/train/
        class1/
            image1.jpg
            image2.jpg
        class2/
            image3.jpg
            etc
"""

train_batches = ImageDataGenerator().flow_from_directory(directory=train_path, target_size=(224,224),
    classes=['dog', 'cat'], batch_size=10)
valid_batches = ImageDataGenerator().flow_from_directory(directory=valid_path, target_size=(224,224),
    classes=['dog', 'cat'], batch_size=10)
test_batches = ImageDataGenerator().flow_from_directory(directory=test_path, target_size=(224,224),
    classes=['dog', 'cat'], batch_size=10)

# Pre-trained model so weights are already stored
# The original trained VGG16 model, along with weights and other parameters, is now downloaded onto our machine.
# First will download the model
vgg16_model = keras.applications.vgg16.VGG16()

# see what the architecture looks like.
# This model is more complex than the Sequential of Convolution (32) -> Flatten (1) -> Dense (2) layer
#vgg16_model.summary()

# the type is not sequential
#type(vgg16_model)

# we will modify the model to move from 1000 categories to only 2 (cats and dogs)
# convert the Functional model to a Sequential model
# we replicate the entire vgg16_model (excluding the output layer) to a new Sequential model named model
# Important - We didn't replicate the last one
model = Sequential()
for layer in vgg16_model.layers[:-1]:
    model.add(layer)

# Freezes the weights and other trainable parameters in each layer.
# They will not be updated when we pass in our images of cats and dogs.
for layer in model.layers:
    layer.trainable = False

# add a layer of 2 outputs
model.add(Dense(2, activation='softmax'))

# print model
model.summary()

# Same compile as in original model
model.compile(optimizer=Adam(learning_rate=.0001), loss='categorical_crossentropy', metrics=['accuracy'])

# Learning on 40 images -> 4 steps and 10 images per step
model.fit_generator(generator=train_batches, steps_per_epoch=4,
    validation_data=valid_batches, validation_steps=4, epochs=5, verbose=2)

test_imgs, test_labels = next(test_batches)
test_labels = test_labels[:,0]

# Taking one batch so will be 10 images
predictions = model.predict_generator(generator=test_batches, steps=1, verbose=0)
print(np.round(predictions[:,0]))

cm = confusion_matrix(y_true=test_labels, y_pred=np.round(predictions[:,0]))
print('Configuration matrix')
print(cm)


"""

Epoch 5/5
4/4 - 39s - loss: 0.4781 - accuracy: 0.8250 - val_loss: 0.5529 - val_accuracy: 0.7750

Configuration matrix
[[5 1]
 [0 4]]
"""