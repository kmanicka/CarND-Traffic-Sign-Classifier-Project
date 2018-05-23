# **Traffic Sign Recognition** 

## Writeup
---

**Build a Traffic Sign Recognition Project**

The goals / steps of this project are the following:
* Load the data set (see below for links to the project data set)
* Explore, summarize and visualize the data set
* Design, train and test a model architecture
* Use the model to make predictions on new images
* Analyze the softmax probabilities of the new images
* Summarize the results with a written report

---

## Rubric Points
### Here I will consider the [rubric points](https://review.udacity.com/#!/rubrics/481/view) individually and describe how I addressed each point in my implementation.  

---
### Writeup / README

#### 1. Provide a Writeup / README that includes all the rubric points and how you addressed each one. You can submit your writeup as markdown or pdf. You can use this template as a guide for writing the report. The submission includes the project code.

#### Overview , 
- LeNet Architecture was used to Classify the German Traffic Signals. 
- The classified achieved 96.5% accuracy on Test Images

Fudther details about this project can be found at [GitHub Project](https://github.com/kmanicka/CarND-Traffic-Sign-Classifier-Project) and [Jypyter Notebook](https://github.com/kmanicka/CarND-Traffic-Sign-Classifier-Project/blob/master/Udacity_Term_1_Traffic_Sign_Classifier.ipynb)

---
### Data Set Summary & Exploration

#### 1. Provide a basic summary of the data set. In the code, the analysis should be done using python, numpy and/or pandas methods rather than hardcoding results manually.

Following are few basic stats about the data set

===============================================
Shape of X_train = (34799, 32, 32, 3)
Shape of X_test = (12630, 32, 32, 3)
Shape of X_valid = (4410, 32, 32, 3)
Shape of Y_train = (34799,)
Shape of Y_test = (12630,)
Shape of Y_valid = (4410,)
===============================================
Number of training examples = 34799
Number of validation examples = 4410
Number of testing examples = 12630
Image data shape = (32, 32, 3)
Number of classes = 43
===============================================
---
#### 2. Include an exploratory visualization of the dataset.

Following are some of the randomly selected images from the dataset. 

![RandomImages](https://raw.githubusercontent.com/kmanicka/CarND-Traffic-Sign-Classifier-Project/master/writeupimages/RandomImages.png)

Verified that the  distribution of the classieded images in train, Validation and Test set are same. 

![Distribution](https://raw.githubusercontent.com/kmanicka/CarND-Traffic-Sign-Classifier-Project/master/writeupimages/distribution.png)

---
### Design and Test a Model Architecture

#### 1. Describe how you preprocessed the image data. What techniques were chosen and why did you choose these techniques? Consider including images showing the output of each preprocessing technique. Pre-processing refers to techniques such as converting to grayscale, normalization, etc. (OPTIONAL: As described in the "Stand Out Suggestions" part of the rubric, if you generated additional data for training, describe why you decided to generate additional data, how you generated the data, and provide example images of the additional data. Then describe the characteristics of the augmented training set like number of images in the set, number of images for each class, etc.)

In the final model, the data input images were normalized and zero meaned using the following formula. 
image = (image - 128) / 128. 

The output classification was converted to one hot format. 

I was able to get some good results without any further pre-processing of the images or data augumentation. 

#### 2. Describe what your final model architecture looks like including model type, layers, layer sizes, connectivity, etc.) Consider including a diagram and/or table describing the final model.

My final model as printed by Keras Model Summary:
_________________________________________________________________
Layer (type)                 Output Shape              Param #   
=================================================================
conv2d_1 (Conv2D)            (None, 28, 28, 32)        2432      
_________________________________________________________________
activation_1 (Activation)    (None, 28, 28, 32)        0         
_________________________________________________________________
max_pooling2d_1 (MaxPooling2 (None, 14, 14, 32)        0         
_________________________________________________________________
conv2d_2 (Conv2D)            (None, 10, 10, 64)        51264     
_________________________________________________________________
activation_2 (Activation)    (None, 10, 10, 64)        0         
_________________________________________________________________
max_pooling2d_2 (MaxPooling2 (None, 5, 5, 64)          0         
_________________________________________________________________
dropout_1 (Dropout)          (None, 5, 5, 64)          0         
_________________________________________________________________
flatten_1 (Flatten)          (None, 1600)              0         
_________________________________________________________________
dense_1 (Dense)              (None, 800)               1280800   
_________________________________________________________________
activation_3 (Activation)    (None, 800)               0         
_________________________________________________________________
dropout_2 (Dropout)          (None, 800)               0         
_________________________________________________________________
dense_2 (Dense)              (None, 128)               102528    
_________________________________________________________________
activation_4 (Activation)    (None, 128)               0         
_________________________________________________________________
dropout_3 (Dropout)          (None, 128)               0         
_________________________________________________________________
dense_3 (Dense)              (None, 43)                5547      
_________________________________________________________________
activation_5 (Activation)    (None, 43)                0         
_________________________________________________________________
Total params: 1,442,571
Trainable params: 1,442,571
Non-trainable params: 0
_________________________________________________________________

#### 3. Describe how you trained your model. The discussion can include the type of optimizer, the batch size, number of epochs and any hyperparameters such as learning rate.

THe base model used was LeNet. Played with following items to get satisfactory results 
1) Number of Filters in the Conv Layers. 
2) Regularization with Dropout
3) Optimizers 
4) Increased Batch Size
5) Normalization of the input data. 

#### 4. Describe the approach taken for finding a solution and getting the validation set accuracy to be at least 0.93. Include in the discussion the results on the training, validation and test sets and where in the code these were calculated. Your approach may have been an iterative process, in which case, outline the steps you took to get to the final solution and why you chose those steps. Perhaps your solution involved an already well known implementation or architecture. In this case, discuss why you think the architecture is suitable for the current problem.

My final model results were:
* training set accuracy of 0.9938
* validation set accuracy of 0.9778 
* test set accuracy of 0.965

I took an iterative approach to tune the model and the hyper parameters. 

#### Iteration 1)  Test Accuracy ~70-80%
As advised in the material, I started off with a LeNet Architecture as base. Used Keras to quickly build a basic LeNet model and trained on the data set. After 5 epocs the model seem to converge and the test accuracy was around 70-80%.

#### Iteration 2) Test Accuracy 90%
This was a good start. As LeNet was using gray scale the inital Conv2D layers with 6 filters were sufficiant. But as we are dealing with color images I tried increasing the number of filters in layer 1 and layer 2  to 32 and 64. This increased the number of trainable parameter output from Layer 2 to 1600. So correspondingly increased the parameters in layer 3 and 4 Dense layers to 800 and 128. Intrestingly this arhitecture gave a very good Train Accuracy of 99% in 10epocs, the Validation and Test Accuracy were arround 90%. 

#### Iteration 3) Test Accuracy -%
There was clearly a variance problem so added few Dropout layers with 0.5 parameter to regularize the model. But the 0.5 model was very high and the model did not converge well. 

#### Iteration 4) Test Accuracy 94%
Tried reducing the dropout to 0.25 and the model performed very well and got a test accuracy of 0.94. 

#### Iteration 5) Test Accuracy 95%
I changed the optimizer to Adadelta() with use an Adaptive Learning Rate Method. This improved the accuracy to 95%. 

#### Iteration 5) Test Accuracy 96.5%
While writing this document i realized that I had not normalized the input image. Once the input images were normalized the test accuracy increased to 96.5%


Following are some of the random images from the training set and the  corresponding classifications predicted by the model. 

![Classification](https://raw.githubusercontent.com/kmanicka/CarND-Traffic-Sign-Classifier-Project/master/writeupimages/classifiedimages.png)



### Test a Model on New Images

#### 1. Choose five German traffic signs found on the web and provide them in the report. For each image, discuss what quality or qualities might be difficult to classify.

Here are 10 German traffic signs that I found on the web:

The image show the origin image, the resized image for input and the softmax output as bar chart. 

![internetpredictions](https://raw.githubusercontent.com/kmanicka/CarND-Traffic-Sign-Classifier-Project/master/writeupimages/internetpredictions1.png)


#### 2. Discuss the model's predictions on these new traffic signs and compare the results to predicting on the test set. At a minimum, discuss what the predictions were, the accuracy on these new predictions, and compare the accuracy to the accuracy on the test set (OPTIONAL: Discuss the results in more detail as described in the "Stand Out Suggestions" part of the rubric).

Here are the results of the prediction:

| Image			        |     Prediction	    | Softmax 				| 
|:---------------------:|:---------------------------------------------:| 
| 50 km/hr      		| 50 km/hr              |   (0.95)	            | 
| 12 t     			    | 120 km/hr             |   (0.92)  			|
| 40 km/hr				| 50 km/hr	            |   (0.20)      		|
| 30 km/hr	      		| 60 km/hr	            |   (0.89)				|
| Pedestrian			| Turn Left Ahead       |   (1.00)   			|
| 14 t			        | Slippery Road         |   (0.38)  			|
| Pedestrian			| General Caution       |   (0.99)              |
| 50 km/hr   			| 50 km/hr              |   (1.00)              |
| 60 km/hr			    | Bycycle Crossing      |   (0.37)         		|

The model was able to correctly guess 4 of the 10 traffic signs, which gives an accuracy of 20%. This compares unfavorably to the accuracy on the test set of 96%. 

I have not tried to manuplate the results and presented the truth as is.  I believe that the key reason of this discrepency is due to the difference in the distribution of the internet images and the data set that we used for training. It is likely that some of the internet images are wrongly labeled or not part of the germen signes at all. 

#### 3. Describe how certain the model is when predicting on each of the five new images by looking at the softmax probabilities for each prediction. Provide the top 5 softmax probabilities for each image along with the sign type of each probability. (OPTIONAL: as described in the "Stand Out Suggestions" part of the rubric, visualizations can also be provided such as bar charts)

The image above shows the histogram of the Softmax prediction for each of he images. The table above show the highest softmax prediction. 

### (Optional) Visualizing the Neural Network (See Step 4 of the Ipython notebook for more details)
#### 1. Discuss the visual output of your trained network's feature maps. What characteristics did the neural network use to make classifications?


