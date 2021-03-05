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
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from torch.utils.data import Dataset, DataLoader, random_split
from PIL import Image
from typing import Tuple
import numpy as np
import argparse
import os

# ----------------- Get Command Line Arguments ------------------------
# construct the argument parser and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-d", "--data_dir", type=str,
                help="path to directory containing image dataset that is to be imported")
ap.add_argument("-m", "--model", type=str, default="HABs_CNN_Model_FINAL.h5",
                help="name of CNN model that will be saved")
args = vars(ap.parse_args())


# ----------------- Create Dataset -------------------------------------
class HABsDataset(Dataset):
    # TODO - not memory efficient because images are all stored in memory first and not read as required
    # TODO - one-hot-encode targets
    # TODO - torchvision's transforms
    # TODO - move HABsDataset to utils
    # TODO - handle train/test inside dataset or outside dataset
    # Referenced https://towardsdatascience.com/building-efficient-custom-datasets-in-pytorch-2563b946fd9f

    def __init__(self, data_root: str):
        self.data_root = data_root
        self.data_samples = []
        self._init_dataset()

    def __len__(self):
        return len(self.data_samples)

    def _init_dataset(self):
        for image_class in os.listdir(self.data_root):
            image_path = os.path.join(self.data_root, image_class)
            image = Image.open(image_path)
            image = self._transform(image)
            target = self._encode_target(image_class)

            self.data_samples.append((image, target))

    @staticmethod
    def _encode_target(target: str) -> int:
        if target == "bga":
            return 0
        elif target == "clear":
            return 1
        elif target == "turbid":
            return 2
        else:
            raise ValueError("Cannot encode. Target must be bga, clear, or turbid.")

    @staticmethod
    def _transform(self, image: Image) -> Image:
        image = np.array(image.resize((32, 32))) / 255.0
        return image

    def __getitem__(self, idx) -> Tuple[Image, int]:
        return self.data_samples[idx]


# ----------------- Load and Split Dataset  --------------------------------
print("\n[INFO] loading images...")

dataset = HABsDataset(args["dataset"])
test_size = int(len(dataset) * 0.75)
train_size = int(len(dataset) * 0.25)
train_data, test_data = random_split(dataset, [test_size, train_size])

train_loader = DataLoader(train_data)
test_loader = DataLoader(test_data)

# TODO - enumerate(train_loader), enumerate(test_loader)

# --------------------- Define Model -----------------------------------
# # define our Convolutional Neural Network architecture
# model = Sequential()
# model.add(Conv2D(8, (3, 3), padding="same", input_shape=(32, 32, 3)))
# model.add(Activation("relu"))
# model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))
# model.add(Conv2D(16, (3, 3), padding="same"))
# model.add(Activation("relu"))
# model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))
# model.add(Conv2D(32, (3, 3), padding="same"))
# model.add(Activation("relu"))
# model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))
# model.add(Flatten())
# model.add(Dense(3))
# model.add(Activation("softmax"))
#
# # ------------------ Compile Model -----------------------------------
# # train the model using the Adam optimizer
# opt = Adam(lr=1e-3, decay=1e-3 / 50)
#
# # compile model
# model.compile(loss="categorical_crossentropy", optimizer=opt,
#               metrics=["accuracy"])
#
# # -------------------  Run Trial -------------------------------------
# print("\n[INFO] training network...")
#
# # fit/train model
# trial = model.fit(trainX, trainY, validation_data=(testX, testY),
#                   epochs=50, batch_size=32)
#
# # get model's prediction
# predictions = model.predict(testX, batch_size=32)
#
# print("\n[INFO] results:")
#
# # get evaluation of model
# evaluation = model.evaluate(testX, testY, verbose=0)
# print("%s = %.2f%%" % (model.metrics_names[1], evaluation[1] * 100))
#
# # get classification report
# classReport = classification_report(testY.argmax(axis=1),
#                                     predictions.argmax(axis=1), target_names=lb.classes_)
# print("\nClassification Report Trial: ")
# print(classReport)
#
# # get confusion matrix
# print("Confusion Matrix Trial: ")
# print(confusion_matrix(testY.argmax(axis=1), predictions.argmax(axis=1)))
#
# # ---------------------- Save Model ------------------------------------
# # save trained model and architecture to single file
# model.save(args["model"])
# print("\nSaved model to disk.\n")
