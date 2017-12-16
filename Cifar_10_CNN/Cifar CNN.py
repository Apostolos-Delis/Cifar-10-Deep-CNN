"""
Created by Apostolos Delis 12/3/17
Convolutional Neural Network that is used to train on the cifar-10 data set that contains 32 by 32 by 3 images in
10 different categories

Training time for this network can take multiple hours if not run on a gpu

Maximum accuracy reached: 76.8%
"""
# imports primarily from keras with a tensorflow backend
import numpy as np
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import tensorflow as tf
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.layers import Conv2D, MaxPooling2D, ZeroPadding2D
from keras.utils import np_utils
from keras.datasets import cifar10
import matplotlib.pyplot as plt
from keras.preprocessing.image import ImageDataGenerator
import time

np.random.seed(101)  # for reproducibility purposes


"""Initialize Neural Network parameters"""

# the batch size of how many images will be processed for each step of stochastic optimization
batch_size = 128

# cifar-10 has 10 classes
nb_classes = 10

# number of epochs the network will iterate through
nb_epoch = 30

# number of neurons in the FC layer
fc_size = 256  # Change to 128 if network is taking too long to train

# data augmentation would increase the training set by attempting to randomly generate new test examples through
# methods such as flipping the images vertically and horizontally or shifting them
data_augmentation = True

# extra parameter on how often during the training process the user should be notified
verbose = 1

# Whether or not to use the deep convolutional neural network
deep = True


"""Initialize the Convolutional layers' hyperparameters"""
# number of convolutional filters to use
nb_filters = 64

# image dimensions
img_rows, img_cols = 32, 32  # Cifar-10 images are 32 by 32 pixels

# convolution kernel size
kernel_size = (3, 3)

# size of pooling area for max pooling
pool_size = (2, 2)

# stride for CNN to take (currently not in use since its value is 1, but if speed is required it can be implemented)
stride = (1, 1)

# zero Padding (add zeros to the tensor boarder in convolutional layers
zero_padding = (1, 1)

# dropout percentage (for regularization)
dropout = 0.4

"""Process the data"""
# split between train and test sets (By default it splits to 50k training and 10k for cross-validation)
(X_train, y_train), (X_test, y_test) = cifar10.load_data()

# Reshape the Data
X_train = X_train.reshape(X_train.shape[0], img_rows, img_cols, X_train.shape[3])  # X_train now (50k, 32, 32, 3)
X_test = X_test.reshape(X_test.shape[0], img_rows, img_cols, X_train.shape[3])  # X_test now (10k, 32, 32, 3)
input_shape = (img_rows, img_cols, X_train.shape[3])  # input shape (32, 32, 3)

# Normalize the data by dividing by the maximum rgb values so that all the values in the tensors are between 0 and 1
X_train = X_train.astype('float32')
X_test = X_test.astype('float32')
X_train /= 255
X_test /= 255

# convert class vectors to binary class matrices, ex: [0, 0, 1, 0, 0, 0]
Y_train = np_utils.to_categorical(y_train, nb_classes)
Y_test = np_utils.to_categorical(y_test, nb_classes)

print("Data Processing complete\nNetwork being initialized...")


"""Create Neural Network"""

if deep:

    conv_filter_scalar = 1  # scalar for how many filters each convolutional layer will have

    model = Sequential()
    model.add(ZeroPadding2D(zero_padding, input_shape=input_shape))
    model.add(Conv2D(nb_filters * conv_filter_scalar, kernel_size=kernel_size, strides=stride))
    model.add(Activation("relu"))
    model.add(ZeroPadding2D(zero_padding))
    model.add(Conv2D(nb_filters * conv_filter_scalar, kernel_size=kernel_size, strides=stride))
    model.add(Activation("relu"))
    model.add(MaxPooling2D(pool_size=pool_size, strides=stride))
    model.add(Dropout(dropout))

    conv_filter_scalar *= 2

    model.add(ZeroPadding2D(zero_padding))
    model.add(Conv2D(nb_filters * conv_filter_scalar, kernel_size=kernel_size, strides=stride))
    model.add(Activation("relu"))
    model.add(ZeroPadding2D(zero_padding))
    model.add(Conv2D(nb_filters * conv_filter_scalar, kernel_size=kernel_size, strides=stride))
    model.add(Activation("relu"))
    model.add(MaxPooling2D(pool_size=pool_size, strides=stride))
    model.add(Dropout(dropout))

    conv_filter_scalar *= 2

    model.add(ZeroPadding2D(zero_padding))
    model.add(Conv2D(nb_filters * conv_filter_scalar, kernel_size=kernel_size, strides=stride))
    model.add(Activation("relu"))
    model.add(ZeroPadding2D(zero_padding))
    model.add(Conv2D(nb_filters * conv_filter_scalar, kernel_size=kernel_size, strides=stride))
    model.add(Activation("relu"))
    model.add(ZeroPadding2D(zero_padding))
    model.add(Conv2D(nb_filters * conv_filter_scalar, kernel_size=kernel_size, strides=stride))
    model.add(Activation("relu"))
    model.add(MaxPooling2D(pool_size=pool_size, strides=stride))
    model.add(Dropout(dropout))

    conv_filter_scalar *= 2

    model.add(ZeroPadding2D(zero_padding))
    model.add(Conv2D(nb_filters * conv_filter_scalar, kernel_size=kernel_size, strides=stride))
    model.add(Activation("relu"))
    model.add(ZeroPadding2D(zero_padding))
    model.add(Conv2D(nb_filters * conv_filter_scalar, kernel_size=kernel_size, strides=stride))
    model.add(Activation("relu"))
    model.add(ZeroPadding2D(zero_padding))
    model.add(Conv2D(nb_filters * conv_filter_scalar, kernel_size=kernel_size, strides=stride))
    model.add(Activation("relu"))
    model.add(MaxPooling2D(pool_size=pool_size, strides=stride))
    model.add(Dropout(dropout))

    conv_filter_scalar /= 2  # start decreasing number of filters depth down to be closer to the number of classes
    conv_filter_scalar = int(conv_filter_scalar)

    model.add(ZeroPadding2D(zero_padding))
    model.add(Conv2D(nb_filters * conv_filter_scalar, kernel_size=kernel_size, strides=stride))
    model.add(Activation("relu"))
    model.add(ZeroPadding2D(zero_padding))
    model.add(Conv2D(nb_filters * conv_filter_scalar, kernel_size=kernel_size, strides=stride))
    model.add(Activation("relu"))
    model.add(ZeroPadding2D(zero_padding))
    model.add(Conv2D(nb_filters * conv_filter_scalar, kernel_size=kernel_size, strides=stride))
    model.add(Activation("relu"))
    model.add(MaxPooling2D(pool_size=pool_size, strides=stride))
    model.add(Dropout(dropout))

    model.add(Flatten())
    model.add(Dense(fc_size * 2))
    model.add(Activation('relu'))
    model.add(Dropout(dropout))
    model.add(Dense(fc_size))
    model.add(Activation('relu'))
    model.add(Dropout(dropout))
    model.add(Dense(nb_classes))
    model.add(Activation('softmax'))

else:

    # This model is not as deep and also does not use zero padding to help speed up the process
    model = Sequential()

    model.add(Conv2D(nb_filters, kernel_size=kernel_size, padding='valid',
                     input_shape=input_shape, strides=stride))
    model.add(Activation("relu"))
    model.add(Conv2D(nb_filters, kernel_size=kernel_size, strides=stride))
    model.add(Activation("relu"))
    model.add(Dropout(dropout))

    model.add(Conv2D(nb_filters, kernel_size=kernel_size, strides=stride))
    model.add(Activation("relu"))
    model.add(Conv2D(nb_filters, kernel_size=kernel_size, strides=stride))
    model.add(Activation("relu"))
    model.add(MaxPooling2D(pool_size=pool_size))
    model.add(Dropout(dropout))

    model.add(Conv2D(nb_filters, kernel_size=kernel_size, strides=stride))
    model.add(Activation("relu"))
    model.add(Conv2D(nb_filters, kernel_size=kernel_size, strides=stride))
    model.add(Activation("relu"))
    model.add(Dropout(dropout))

    model.add(Conv2D(nb_filters, kernel_size=kernel_size, strides=stride))
    model.add(Activation("relu"))
    model.add(Conv2D(nb_filters, kernel_size=kernel_size, strides=stride))
    model.add(Activation("relu"))
    model.add(MaxPooling2D(pool_size=pool_size))
    model.add(Dropout(dropout))

    model.add(Flatten())
    model.add(Dense(fc_size))
    model.add(Activation('relu'))
    model.add(Dropout(dropout))
    model.add(Dense(nb_classes))
    model.add(Activation('softmax'))


model.compile(loss='categorical_crossentropy',
              optimizer='adam',  # as of 12/3/17, adam was the best stochastic optimization technique
              metrics=['accuracy'])

start = time.time()
"""Determine if and how data will be augmented """
if data_augmentation:

    print('Using real-time data augmentation.')

    # This will do pre-processing and real-time data augmentation:
    data_generator = ImageDataGenerator(
        featurewise_center=False,  # set input mean to 0 over the dataset
        samplewise_center=False,  # set each sample mean to 0
        featurewise_std_normalization=False,  # divide inputs by std of the dataset
        samplewise_std_normalization=False,  # divide each input by its x`x`
        zca_whitening=False,  # apply whitening
        rotation_range=0,  # randomly rotate images in the range (degrees, 0 to 180)
        width_shift_range=0.1,  # randomly shift images horizontally (fraction of total width)
        height_shift_range=0.1,  # randomly shift images vertically (fraction of total height)
        horizontal_flip=True,  # randomly flip images horizontally
        vertical_flip=False)  # randomly flip images vertically

    data_generator.fit(X_train)

    # Fit the model on the batches generated by datagen.flow().
    history = model.fit_generator(data_generator.flow(X_train, Y_train, batch_size=batch_size), workers=4,
                                  steps_per_epoch=X_train.shape[0], nb_epoch=nb_epoch,
                                  validation_data=(X_test, Y_test), verbose=verbose)

else:
    print('Using real-time data augmentation.')

    history = model.fit(X_train, Y_train, batch_size=batch_size, nb_epoch=nb_epoch, shuffle=True,
                        verbose=verbose, validation_data=(X_test, Y_test))  # set verbose to 0 for faster training

print("Training Complete")
end = time.time()
print("Model took %0.2f seconds to train" % (end - start))


"""Start Generating Metrics"""

timestr = time.strftime("%Y-%m-%d_%H-%M-%S")

# Plot the Loss Curves
plt.figure(figsize=[8, 6])
plt.plot(history.history['loss'], 'r', linewidth=3.0)
plt.plot(history.history['val_loss'], 'b', linewidth=3.0)

plt.legend(['Training loss', 'Validation Loss'], fontsize=18)
plt.xlabel('Epochs ', fontsize=16)
plt.ylabel('Loss', fontsize=16)
plt.title('Loss Curves', fontsize=16)

# Save the Graph png and overwrite the latest graph
loss_curve_name = "Loss_Graph_" + timestr + ".png"
latest_curve_name = "Latest_Loss_Graph.png"
plt.savefig(loss_curve_name)
plt.savefig(latest_curve_name)


# Plot the Accuracy Curves
plt.figure(figsize=[8, 6])
plt.plot(history.history['acc'], 'r', linewidth=3.0)
plt.plot(history.history['val_acc'], 'b', linewidth=3.0)

plt.legend(['Training Accuracy', 'Validation Accuracy'], fontsize=18)
plt.xlabel('Epochs ', fontsize=16)
plt.ylabel('Accuracy', fontsize=16)
plt.title('Accuracy Curves', fontsize=16)

# Save the Graph png and overwrite the latest graph
accuracy_curve_name = "Accuracy_Graph_" + timestr + ".png"
latest_accuracy_curve_name = "Latest_Accuracy_Graph.png"
plt.savefig(accuracy_curve_name)
plt.savefig(latest_accuracy_curve_name)

print("Plotting complete")

score = model.evaluate(X_test, Y_test, verbose=0)
print('Test score:', score[0])
print('Test accuracy:', score[1])


"""Save the Model to json"""

# serialize model to JSON
model_json = model.to_json()
with open("CNN_model.json", "w+") as json_file:
    json_file.write(model_json)
    json_file.close()

# serialize weights to HDF5
model.save_weights("CNN_model_" + timestr + ".h5")
model.save_weights("Latest_CNN.h5")
print("Saved model to disk")
