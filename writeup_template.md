#**Behavioral Cloning** 

---

**Behavioral Cloning Project**

The goals / steps of this project are the following:

* Use the simulator to collect data of good driving behavior
* Build, a convolution neural network in Keras that predicts steering angles from images
* Train and validate the model with a training and validation set
* Test that the model successfully drives around track one without leaving the road

[//]: # (Image References)

[image1]: ./output_image/center.jpg
[image2]: ./output_image/left.jpg
[image3]: ./output_image/right.jpg
[image4]: ./output_image/output.png
[video1]: ./output_video.mp4

## Rubric Points
Here I will consider the [rubric points](https://review.udacity.com/#!/rubrics/432/view) individually and describe how I addressed each point in my implementation.  

---

My project includes the following files:

* model.py containing the script to create and train the model
* drive.py for driving the car in autonomous mode
* model.h5 containing a trained convolution neural network 
* writeup_report.md or writeup_report.pdf summarizing the results

## How to use Udacity simulator
Using the Udacity provided simulator and my drive.py file, the car can be driven autonomously around the track by executing 

```sh
python drive.py model.h5
```

***

## Model Architecture and Training Strategy

####1. An appropriate model architecture has been employed
Now I adopted nvidia model. Network is expressed as follows:

|Layer (type)|	Output Shape|	Params|	Connected to|
|:-----------|--------------|---------|-------------|
|lambda_1 (Lambda)|	(None, 66, 200, 3)|	0|	lambda_input_1|
|convolution2d_1 (Convolution2D)|	(None, 31, 98, 24)	|1824	|lambda_1|
|convolution2d_2 (Convolution2D)|	(None, 14, 47, 36)	|21636	|convolution2d_1|
|convolution2d_3 (Convolution2D)|	(None, 5, 22, 48)	|43248	|convolution2d_2|
|convolution2d_4 (Convolution2D)|	(None, 3, 20, 64)	|27712	|convolution2d_3|
|convolution2d_5 (Convolution2D)|	(None, 1, 18, 64)	|36928	|convolution2d_4|
|flatten_1 (Flatten)|	(None, 1152)|	0	|dropout_1|
|dense_1 (Dense)|	(None, 100)	|115300	|flatten_1|
|dense_2 (Dense)|	(None, 50)	|5050	|dense_1|
|dense_3 (Dense)|	(None, 10)	|510	|dense_2|
|dense_4 (Dense)|	(None, 1)	|11	|dense_3|


####2. Attempts to reduce overfitting in the model
In order to reduce overfitting, I used data augmentation.
Multi-view camera is used, and mirrored image is created as below:

```
name = './data/IMG/'+batch_sample[0].split('/')[-1]
center_image = cv2.imread(name)
center_angle = float(batch_sample[3])
name = './data/IMG/'+batch_sample[1].split('/')[-1]
left_image = cv2.imread(name)
left_angle = float(batch_sample[3]) + correction
name = './data/IMG/'+batch_sample[2].split('/')[-1]
right_image = cv2.imread(name)
right_angle = float(batch_sample[3]) - correction

inverse_center_image = cv2.flip(center_image, 1)
inverse_center_angle = center_angle*-1.0
inverse_left_image = cv2.flip(left_image, 1)
inverse_left_angle = left_angle*-1.0
inverse_right_image = cv2.flip(right_image, 1)
inverse_right_angle = right_angle*-1.0
```

###### Center Camera
![alt text][image1]

###### Left Camera
![alt text][image2]

###### Right Camera
![alt text][image3]


####3. Model parameter tuning

The model used an adam optimizer, so the learning rate was not tuned manually.

####4. Appropriate training data
I used pre-captured dataset provided by Udacity.

###Model Architecture and Training Strategy

####1. Solution Design Approach
I did some data augmentation.
But it seems still seems to have overfitting.

![alt text][image4]


I need some improvents for this.

####2. Output Movie
![alt text][video1]
Here's a [link to my video result](./output.mp4). I'm afraid this still needs to be improved.