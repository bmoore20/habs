# Author: Bizzy Moore
# Hobart and William Smith Colleges
# CPSC 450 - Independent Study
# Spring 2020

# ----------------- Program Notes -----------------------------
# Classifies images of the Finger Lakes into three categories:
# blue-green algae (BGA), clear, or turbid.
# Sorts the images into a folder according to the specific
# classification that they received.
# Produces a report that supplies information about how the
# model classified each image.

# --------------- Import Libraries ---------------------------
# import the necessary libraries and packages
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
from keras.models import load_model
import shutil
import platform
from pathlib import PurePosixPath
from pathlib import PureWindowsPath
from operator import itemgetter
from keras.preprocessing.image import load_img
from keras.preprocessing.image import img_to_array

# --------------- Determine Operating System ---------------------
# get computer's operating system
opSys = platform.system()

# get seperator based on OS
# set OS flags
if opSys == 'Linux':  # OS is Linux
    seperator = '/'
    isLinux = True
    isMac = False
    isWindows = False

if opSys == 'Darwin':  # OS is Mac
    seperator = '/'
    isLinux = False
    isMac = True
    isWindows = False

if opSys == 'Windows':  # OS is windows
    seperator = '\\'
    isLinux = False
    isMac = False
    isWindows = True

# --------------- Get Command Line Arguments  -----------------------------
# construct the argument parser and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-d", "--dataset", type=str, default="ImportedPics",
                help="path to directory containing image dataset that is to be imported")
ap.add_argument("-c", "--classpics", type=str, default="ClassifiedPics",
                help="name of the folder that holds the classified pictures")
ap.add_argument("-t", "--txtfilename", type=str, default="HABsClassReport",
                help="name of the text file that will be created")
ap.add_argument("-m", "--model", type=str, default="HABs_CNN_Model_FINAL.h5",
                help="name of CNN model that is loaded")
args = vars(ap.parse_args())

# --------------- Load Model ------------------------------
print("\n[INFO] loading model...\n")

# load pre-trained model into program
model = load_model(args["model"])

# ------------------ Load Dataset -----------------------
print("\n[INFO] loading images...\n")

# grab all image paths in the input dataset directory
imagePaths = paths.list_images(args["dataset"])

# initialize lists of source directories and images
imageSRCs = []
data = []

# loop over our input images
for imagePath in imagePaths:
    # add image's path to source directory list
    imageSRCs.append(imagePath)

    # load the input image from disk
    image = load_img(imagePath)

    # reside the image to 32x32 pixels
    imageResize = image.resize((32, 32))

    # convert image to a numpy array
    imageToArray = img_to_array(imageResize)

    # scale image to the range [0,1]
    # divide by 255 because 255 is the max rgb value for each pixel, 0-255
    imageDivide = imageToArray / 255.0

    # expand the dimensions of image
    imageExpand = np.expand_dims(imageDivide, axis=0)

    # add the image to the data list
    data.append(imageExpand)

# put the complete data list in a vstack
dataVStack = np.vstack(data)

# ---------- Classify the Images --------------
print("[INFO] classifying images...\n")

# generate model's classification predictions for each input image
# outputs array for each image with the % of confidence for each classification label
predictions = model.predict(dataVStack)

# gives us the model's classification for each input image
# takes the index that contains the max % value from each image's predictions array
classifications = predictions.argmax(axis=1)

# ----------- Save Calssified Images in Sorted Folders -----------
print("[INFO] sorting images...\n")

# create the main list that will hold all of the images' classification information
totalList = []

# copy each image into the correct folder based on their new classification
for imageSRC, predictedNums, classNum in zip(imageSRCs, predictions, classifications):
    # get filename of image
    if isWindows == False:
        # use PurePosixPath for Mac and Linux os
        filenameImg = PurePosixPath(imageSRC).name
    if isWindows == True:
        # use PureWindowsPath
        filenameImg = PureWindowsPath(imageSRC).name

    # get max % value from images predicted array
    maxVal = max(predictedNums)

    # convert value to a string
    strMaxVal = str(maxVal)

    # get the first 4 decimal places from the max % value
    # will be used to help sort the images based on the model's confidence level
    maxSorter = strMaxVal[2:6]

    # get the classification folder and name for image
    if classNum == 0:
        # change path to bga folder
        classifiedFolder = "bgaClassified"
        className = 'bga'
    elif classNum == 1:
        # change path to clear folder
        classifiedFolder = "clearClassified"
        className = 'clear'
    elif classNum == 2:
        # change path to turbid folder
        classifiedFolder = "turbidClassified"
        className = 'turbid'
    else:
        # set to miscellaneous
        classifiedFolder = "miscellaneous"
        className = 'miscell'

    # create list with classification information for image
    listCluster = [maxSorter, filenameImg, className]

    # add the predicted numbers for each category to the list
    for num in predictedNums:
        listCluster.append(num)

    # add images list cluster to the main list
    totalList.append(listCluster)

    # create a new filename that the image will be saved as in its classified folder
    # this new filename contains the maxSorter
    # this allows the images to be sorted by confidence level in its classified folder
    newFilename = maxSorter + '-' + filenameImg

    # get the name of the folder that holds all of the classified images
    classifiedpicsfolder = args["classpics"]

    # create the new file destination for the image
    # make sure that the correct current directory (cd) is set
    # goes to relative path of a file in a subdirectory of the cd
    dstFile = classifiedpicsfolder + seperator + classifiedFolder + seperator + newFilename

    # copy file into correct classification folder
    shutil.copy(imageSRC, dstFile, follow_symlinks=True)

# ------------------- Generate Report -------------------------------
print("[INFO] generating report...\n")

# sort the list clusters that are in totalList in ascending order
totalList = sorted(totalList, key=itemgetter(0))

# create name for report
textfilename = args["txtfilename"] + ".txt"

# write to report file
file = open(textfilename, "w")
file.write(",MaxValueSorter,ImageName,Classification,bgaConfidence,clearConfidence,turbidConfidence\n")

# write each list cluster from total list into report
for cluster in totalList:
    clusterString = ""
    for thing in cluster:
        clusterString = clusterString + "," + str(thing)
    clusterString = clusterString + "\n"
    file.write(clusterString)
file.close()

print("Program complete. \n")
