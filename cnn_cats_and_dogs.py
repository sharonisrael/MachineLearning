import os
import numpy as np
import matplotlib.pyplot as plt

from  tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Sequential
from tensorflow_core.python.keras.layers import Dense, Conv2D, Flatten
from tensorflow.keras.optimizers import Adam

from sklearn.metrics import confusion_matrix

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

train_files_dict = {'cat' : len(os.listdir(train_path + 'cat')), 'dog' : len(os.listdir(train_path + 'dog'))}
print(train_files_dict)
#fig = plt.figure()
#ax = fig.add_axes([0,0,1,1])
#ax.bar(files_dict.keys(), files_dict.values())
#plt.show()

print(train_path)
train_batches = ImageDataGenerator().flow_from_directory(directory=train_path, target_size=(224,224),
    classes=['dog', 'cat'], batch_size=10)
valid_batches = ImageDataGenerator().flow_from_directory(directory=valid_path, target_size=(224,224),
    classes=['dog', 'cat'], batch_size=10)
test_batches = ImageDataGenerator().flow_from_directory(directory=test_path, target_size=(224,224),
    classes=['dog', 'cat'], batch_size=10)

# show first train images
#imgs, labels = next(train_batches)
#plt.imshow(np.array(imgs).astype(np.uint8)[0])
#plt.show()
#im = Image.open(train_path+sample)  # base = Image.open('Pillow/Tests/image.png').convert('RGBA')
#im.show()

# plots images with labels within jupyter notebook
def plots(ims, figsize=(12,6), rows=1, interp=False, titles=None):
    if type(ims[0]) is np.ndarray:
        ims = np.array(ims).astype(np.uint8)
        if (ims.shape[-1] != 3):
            ims = ims.transpose((0,2,3,1))
    f = plt.figure(figsize=figsize)
    cols = len(ims)//rows if len(ims) % 2 == 0 else len(ims)//rows + 1
    for i in range(len(ims)):
        sp = f.add_subplot(rows, cols, i+1)
        sp.axis('Off')
        if titles is not None:
            sp.set_title(titles[i], fontsize=16)
        plt.imshow(ims[i], interpolation=None if interp else 'none')

#  one-hot encoding [x,y] = [cat,dog] ==> of [1, 0], and cats are represented by [0,1]
#plots(imgs, titles=labels)
#plt.show()

model = Sequential([
    #  2-dimensional convolutional layer because of 2D image
    # You can experiment by choosing different values for these parameters.
    # 32 output filters
    # Kernel of 3*3
    # input_shape is resizing the image so it will take less time. FOr example to 64*64
    Conv2D(filters=32, kernel_size=(3, 3), activation='relu', input_shape=(224,224,3)),
    # We’ll then Flatten the output from the convolutional layer and pass it to a Dense layer
    # Take previous layer to 1d tensor
    Flatten(),
    # Output for 2 options: cats or dogs
    Dense(2, activation='softmax'),
    ])

model.summary()

model.compile(optimizer=Adam(learning_rate=0.0001), loss='categorical_crossentropy', metrics=['accuracy'])

model.fit_generator(generator=train_batches, steps_per_epoch=4,
    validation_data=valid_batches, validation_steps=4, epochs=5, verbose=2)

test_imgs, test_labels = next(test_batches)
# show images with encoding
#plots(test_imgs, titles=test_labels)
#plt.show()

# show labels
#print("test lables table")
#print(test_labels)
# It is a notation used in Numpy/Pandas.
# [ : , 0 ] means [ first_row:last_row , column_0 ]. If you have a 2-dimensional list/matrix/array,
# this notation will give you all the values in column 0 (from all rows).
# Here it's relavent because of one hot wire so can be their 0 or 1
print("Array of test labels")
test_labels = test_labels[:,0]
print(test_labels)

# before it was fit to build the model
# now it's predict to check te test values
# steps = N it means will do N-1 batch steps. The bach size is in ImageDataGenerator.
# so (batch_size = 10, steps = 4) = 31 images
predictions = model.predict_generator(generator=test_batches, steps=1, verbose=0)
#print("predictions table")
#print(predictions)
"""
    output is something like
    [[1.0000000e+00 0.0000000e+00]
     [1.0000000e+00 0.0000000e+00]
     [1.0000000e+00 7.4927429e-42]]
"""

# Taking first numpy column
print("Array of predictions labels")
print(np.round(predictions[:,0]))

print("Array of true labels")
print(test_batches.class_indices)

# Creating confusion matrix of 2 lists -> test_labels and predictions
cm = confusion_matrix(y_true=test_labels, y_pred=np.round(predictions[:,0]))
print('Configuration matrix')
print(cm)
"""
Array of test labels
[1. 0. 1. 1. 0. 0. 1. 0. 0. 0.]
Array of predictions labels
[0. 0. 0. 0. 0. 0. 0. 0. 0. 0.]

The prediction is only cats

real cat 
real dog
          model cat  model dog

real cat  6 
real dog  4
          model cat  model dog

[[6 0]
 [4 0]]
 
model correctly predicted image was a cat 6 times when it actually was a cat
and incorrectly predicted image was a cat 5 times when it was not a cat. 
At this point, the model is no better than chance. 
"""