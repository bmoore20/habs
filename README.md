# Harmful Agal Blooms

![614238A6-E71E-4EEB-80AB-007F44976D75_1_105_c](https://user-images.githubusercontent.com/67593919/128557794-7c31e948-bce0-41a5-8074-3f9affc13e34.jpeg)

Harmful Algal Blooms (HABs) are a critical problem that have been negatively affecting bodies of water across the globe. *habs* utilizes machine learning to identify and classify HABs in order to help the effort of improving how to track and predict blooms.

## Project History ## 

![C3C95CEC-C3EA-424C-8DED-A4BB458F696B_1_105_c](https://user-images.githubusercontent.com/67593919/128559955-8819a77b-0b4b-4476-9937-1a55c39914a6.jpeg)

This project originated in the spring of 2020 when I completed an independent study with the Computer Science and Physics departments at Hobart and William Smith Colleges (HWS). After graduating in May 2020, I set the project aside for a little while. During the winter of 2021, I decided to revisit the project with the goal of improving the accuracy of the model and refactoring the code base. 

The original implementation trained the model with Keras. It then used the trained model to classify new images and sort them into folders based on their classification. In addition, the script generated a report that provided details about the results for each image prediction. 

For the revamped version of *habs*, I switched over to PyTorch for the machine learning implementation and restructured the whole repository. 

This site will follow the progress of the improved version of *habs*. It is found in `main` branch of the **bmoore20/habs** repository. The code for my initial implementation of this project can be found under the branch named `original`.

## Data ##

In recent years, HABs have been frequenting the Finger Lakes Region of New York State. The image data used to train the model was collected by professors and researchers at HWS and the Finger Lakes Institute in Geneva, NY. All of the images were taken at the same location from the same height. This consistency limits bias when we are training our model.

The biggest challenge for this project is the lack of data we have for the HABs. We only have a handful of good examples of HAB images. However, in order to successfully train a machine learning model, you need thousands of images for each class. Therefore, we had to perform various methods of data augmentation. These included applying different combinations of transformations, oversampling, and transfer learning. 

There are three different types of image data: blue-green algae (bga), clear, and turbid. The blue-green algea images portray instances of HABs. The clear and turbid images illustrate what the water looks like when it is healthy. We give the model examples of both algae and non-algae images because we ultimately want it to be able to detect the differences between these various states. 

##### Examples of BGA Images #####
![bgaclear00127](https://user-images.githubusercontent.com/67593919/128363237-0c73c731-466c-4dba-b221-4a204c0f7159.jpg)

![bgaclear00107](https://user-images.githubusercontent.com/67593919/128558098-0549a9b7-3bfb-4c15-989e-dfbd64478275.jpg)

##### Example of Clear Image #####
![MP 09-03-19 x00069](https://user-images.githubusercontent.com/67593919/128557501-fb0e915a-8855-4018-9cfc-a0f7f4443040.jpg)

##### Example of Turbid Image #####
![MP 09-02-19 x00116](https://user-images.githubusercontent.com/67593919/128557709-4dfa6f1f-1e63-44e1-acb9-dbbc7d92f4df.jpg)

## About the Author ##

My name is Elizabeth Moore and I am from Buffalo, NY. I graduated from HWS in 2020 with a major in Computer Science and a minor in Physics. I am interested in the fields of ML, AI, and DS and love solving challenging problems. 

## Acknowledgements ##

A big thank you to Tynan Daly, Ileana Dumitriu (HWS), Stina Bridgeman (HWS), and John Halfman (HWS) for their help, guidance, and support throughout this project. 

The images of the Finger Lakes were taken by a fellow HWS classmate, Adam Farid. 

![60CD77FB-540F-4ECE-A59F-F9987E2A3B16_1_105_c](https://user-images.githubusercontent.com/67593919/128569040-8ba150be-f642-4d61-b9f9-025ca5f80097.jpeg)

