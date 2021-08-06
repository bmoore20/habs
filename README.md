# HABs #
Detect Harmful Algal Blooms (HABs) in images of the Finger Lakes.

HABs are a critical problem that have been negatively affecting bodies of water across the globe. *habs* utilizes 
machine learning to identify and classify HABs in order to help the effort of improving how to track and predict 
blooms. In recent years, HABs have been frequenting the Finger Lakes Region of New York State. The image data used to
train the model was collected by professors and researchers at Hobart and William Smith Colleges (HWS) and the Finger 
Lakes Institute in Geneva, NY. 

##### HABs Website #####

https://bmoore20.github.io/habs/

### Installation and Dependencies: ###
You can create an environment that contains all of the dependencies needed to run *habs* via Conda. 

After cloning the repository, navigate to the directory where you saved it on your local machine and run the following:

```
conda env create -f environment.yml
```

This will create a new environment called `habs-env`.

### Running Program: ###
##### Colab #####
Colab can be used to run the program. See the link below for an example on how to run *habs* in Colab. 
 
https://colab.research.google.com/drive/1BMy11klxZctS0RkHYPxUdCnwnQgvt2FB?usp=sharing

##### TensorBoard #####

TensorBoard is used to help visualize the results of the model throughout the training process. See the link below for 
an example on how to run TensorBoard with *habs*. 

https://colab.research.google.com/drive/1Dieovj38izKeh94bpZgXNXn_DXmSaqhP?usp=sharing

##### Directory Setup ##### 

In order to train the model, the image files have to be organized in a particular way. This is because the folder names
that the images are in act as the label for the images. Therefore, it is very important that the directory paths for 
the images are set up correctly. See the link below for example directories. 

https://drive.google.com/drive/folders/1G9nkZ-Nz8KeGzS_gADPru4NLdk9MbOZN?usp=sharing 


### Background Information: ###

##### HABs Paper #####

https://drive.google.com/drive/folders/1N2CFbhTRR9yqlSB67EDZuQrzYO-QarEG?usp=sharing




