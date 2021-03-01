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
# from sklearn.preprocessing import LabelBinarizer
# from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from PIL import Image
# from imutils import paths
from typing import Tuple
from pathlib import Path
import numpy as np
import argparse
import os
import torch

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


class HABsDataset(torch.utils.data.Dataset):
    # TODO - move to utils
    # TODO - train vs test datasets
    # TODO - torchvision -> transforms.Compose()
    def __int__(self, data_dir: str, transform=None):
        """Set of images from the Finger Lakes that consist of harmful algal blooms.

        :param data_dir: file path to root of dataset directory
        :param transform: optional transform to be applied to a sample
        :return:
        """
        self.data_dir = data_dir

    def __len__(self):
        # number of images in directory
        return len(os.listdir(self.data_dir))

    def _transform_rescale(self, image: Image) -> Image:
        """Resize image to 32x32 pixels.
           Scale the pixel intensities to the range [0,1]
           Divide by 255 because 255 is the max rgb value for each pixel, 0-255

        :param image: image to be scaled
        :return: rescaled image
        """
        image = np.array(image.resize((32, 32))) / 255.0
        return image

    def __getitem__(self, idx) -> Tuple[Image, int]:
        """Retrieve the ith sample of the dataset.

        :param idx: the index of the sample to be retrieved
        :return: the transformed image and its target in number form
        """
        # TODO - use pathlib.Path instead of imutils.paths -> imutils.paths is a 'generator' object
        # TODO - get path_to_data
        # TODO - make image names uniform with idx at end
        # TODO - extract string label from path_to_data and convert to number representation (target)
        # TODO - apply transforms to image
        # TODO - return tuple of transformed image and target
        ...

# TODO - use DataLoader -> torch.utils.data.DataLoader

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
