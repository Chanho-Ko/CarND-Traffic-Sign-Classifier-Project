# **Traffic Sign Recognition** 

## Overview

In this project, deep neural networks and convolutional neural networks to classify traffict signs are implemented with tensorflow in phython. Specifically, traffic sign images from [German Traffic Sign Dataset](http://benchmark.ini.rub.de/?section=gtsrb&subsection=dataset) are trained to classify.

The main source code can be found in the [Traffic_Sign_Classifier_CHK.ipynb](./Traffic_Sign_Classifier_CHK.ipynb). Launch this file with Jupyter notebook.

**Pipeline of Traffic Sign Recognition Project**

The goals / steps of this project are the following:

* Load the data set
* Explore, summarize and visualize the data set
* Design, train and test a model architecture
* Use the model to make predictions on new images
* Analyze the softmax probabilities of the new images


[//]: # "Image References"

[image1]: ./examples/visualization.png "Visualization"
[image2]: ./examples/grayscale.jpg "Grayscaling"
[image3]: ./examples/random_noise.jpg "Random Noise"
[image4]: ./examples/placeholder.png "Traffic Sign 1"
[image5]: ./examples/placeholder.png "Traffic Sign 2"
[image6]: ./examples/placeholder.png "Traffic Sign 3"
[image7]: ./examples/placeholder.png "Traffic Sign 4"
[image8]: ./examples/placeholder.png "Traffic Sign 5"

---

### Step 1: Data Set Summary & Exploration

#### 1. Summary of data set

Image file from data set is a *pickle* file. I used `import pickle` package for loading pickled data.

* The size of training set: **34799**
* The size of the validation set: **4410**
* The size of test set: **12630**
* The shape of a traffic sign image: **(32, 32, 3)**
* The number of unique classes/labels in the data set: **43**

#### 2. Include an exploratory visualization of the dataset.

This part shows random image from the training set and corresponding class number. Using pandas library, the corresponding sign name can be printed by reading [csv file](./signnames.csv).

The example output of exploratory visualization of a ramdom traffic sign image is as follow:

![alt text][image1]

---

### Step 2: Design and Test a Model Architecture

#### 1. Pre-processing

I just suffled the dataset before the training.




#### 2. Model Architecture

My final model consisted of the following layers:

|      Layer      |                 Description                 |
| :-------------: | :-----------------------------------------: |
|      Input      |              32x32x3 RGB image              |
| Convolution 5x5 | 1x1 stride, valid padding, outputs 14x14x6  |
|      RELU       |                                             |
|   Max pooling   |        2x2 stride,  outputs 16x16x6         |
| Convolution 5x5 | 1x1 stride, valid padding, outputs 10x10x16 |
|      RELU       |                                             |
|   Max pooling   |         2x2 stride,  outputs 5x5x16         |
| Fully connected |                 outputs 120                 |
|      RELU       |                                             |
| Fully connected |                 outputs 84                  |
|      RELU       |                                             |
| Fully connected |                 outputs 43                  |



#### 3. Describe how you trained your model. The discussion can include the type of optimizer, the batch size, number of epochs and any hyperparameters such as learning rate.

To train the model, I used 40 epochs and 128 batchs. The learning rate is set to 0.001. I used optimizer of `tf.train.AdamOptimizer`.

#### 4. Describe the approach taken for finding a solution and getting the validation set accuracy to be at least 0.93. Include in the discussion the results on the training, validation and test sets and where in the code these were calculated. Your approach may have been an iterative process, in which case, outline the steps you took to get to the final solution and why you chose those steps. Perhaps your solution involved an already well known implementation or architecture. In this case, discuss why you think the architecture is suitable for the current problem.

My final model results were:

* training set accuracy of **0.994**

* validation set accuracy of **0.885**

* test set accuracy of **0.867**

  

---



### Step 3: Test a Model on New Images

#### 1. Choose five German traffic signs found on the web.



#### 2. Predict the sign type for each image

Here are the results of the prediction:

|        Image         |      Prediction      |
| :------------------: | :------------------: |
|     No vehicles      |     No vehicles      |
|        Yield         |        Yield         |
|      Keep right      |      Keep right      |
| Speed limit (30km/h) | Speed limit (30km/h) |
|      Keep right      |      Keep right      |


The model was able to correctly guess 5 of the 5 traffic signs, which gives an accuracy of 100%. 
