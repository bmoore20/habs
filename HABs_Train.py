# Author: Bizzy Moore
# Hobart and William Smith Colleges
# CPSC 450 - Independent Study
# Spring 2020

# ----------------- Program Notes -----------------------------
# Trains and saves a Convolutional Neural Network (CNN)
# using the Keras library.
# The model is trained using images from the Finger Lakes.

# --------------- Import Libraries ---------------------------
from keras.models import Sequential
from keras.layers.convolutional import Conv2D
from keras.layers.convolutional import MaxPooling2D
from keras.layers.core import Activation
from keras.layers.core import Flatten
from keras.layers.core import Dense
from keras.optimizers import Adam
from sklearn.preprocessing import LabelBinarizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from PIL import Image
from imutils import paths
import numpy as np
import argparse
import os

# ----------------- Get Command Line Arguments ------------------------
# construct the argument parser and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-d", "--dataset", type=str, default="ImportedPics",
                help="path to directory containing image dataset that is to be imported")
ap.add_argument("-m", "--model", type=str, default="HABs_CNN_Model_FINAL.h5",
                help="name of CNN model that will be saved")
args = vars(ap.parse_args())

# ----------------- Load Dataset --------------------------------------
print("\n[INFO] loading images...")

# grab all image paths in the input dataset directory
# initialize lists of images and corresponding class labels
imagePaths = paths.list_images(args["dataset"])
data = []
labels = []

# loop over our input images
for imagePath in imagePaths:
    # load the input image from disk
    # resize it to 32x32 pixels
    # scale the pixel intensities to the range [0,1]
    # divide by 255 because 255 is the max rgb value for each pixel, 0-255
    # update the image list
    image = Image.open(imagePath)
    image = np.array(image.resize((32, 32))) / 255.0
    data.append(image)

    # extract the class label from the file path
    # update the labels list
    label = imagePath.split(os.path.sep)[-2]
    labels.append(label)

# encode the labels, converting them from strings to integers
lb = LabelBinarizer()
labels = lb.fit_transform(labels)

# -------------------- Split Data -------------------------------------
# split data set into a training set a test set
(trainX, testX, trainY, testY) = train_test_split(np.array(data),
                                                  np.array(labels), test_size=0.10)

# --------------------- Define Model -----------------------------------
# define our Convolutional Neural Network architecture
model = Sequential()
model.add(Conv2D(8, (3, 3), padding="same", input_shape=(32, 32, 3)))
model.add(Activation("relu"))
model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))
model.add(Conv2D(16, (3, 3), padding="same"))
model.add(Activation("relu"))
model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))
model.add(Conv2D(32, (3, 3), padding="same"))
model.add(Activation("relu"))
model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))
model.add(Flatten())
model.add(Dense(3))
model.add(Activation("softmax"))

# ------------------ Compile Model -----------------------------------
# train the model using the Adam optimizer
opt = Adam(lr=1e-3, decay=1e-3 / 50)

# compile model
model.compile(loss="categorical_crossentropy", optimizer=opt,
              metrics=["accuracy"])

# -------------------  Run Trial -------------------------------------
print("\n[INFO] training network...")

# fit/train model
trial = model.fit(trainX, trainY, validation_data=(testX, testY),
                  epochs=50, batch_size=32)

# get model's prediction
predictions = model.predict(testX, batch_size=32)

print("\n[INFO] results:")

# get evaluation of model
evaluation = model.evaluate(testX, testY, verbose=0)
print("%s = %.2f%%" % (model.metrics_names[1], evaluation[1] * 100))

# get classification report
classReport = classification_report(testY.argmax(axis=1),
                                    predictions.argmax(axis=1), target_names=lb.classes_)
print("\nClassification Report Trial: ")
print(classReport)

# get confusion matrix
print("Confusion Matrix Trial: ")
print(confusion_matrix(testY.argmax(axis=1), predictions.argmax(axis=1)))

# ---------------------- Save Model ------------------------------------
# save trained model and architecture to single file
model.save(args["model"])
print("\nSaved model to disk.\n")
