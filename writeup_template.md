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

#### 1. An appropriate model architecture has been employed
At first, I adopted nvidia model. Because the situation is quite similar to what we want to do.
But this model goes overfitting because of few amount of data.

So I tried to reduce complexity of the network:

* Fully connected layer is reduced from 4 to 3.
* Dropout layer is added from original.

Final network architecture I used is expressed as follows:

|Layer (type)|	Output Shape|	Params|	Connected to|
|:-----------|--------------|---------|-------------|
|lambda_1 (Lambda)|	(None, 160, 320, 3))|	0|	lambda_input_1|
|cropping2d_1 (Cropping2D) |  (None, 65, 320, 3)|0	|lambda_1|
|convolution2d_1 (Convolution2D) |  (None, 31, 158, 24)|1824	|cropping2d_1|
|convolution2d_2 (Convolution2D)|	(None, 14, 77, 36)	|21636	|convolution2d_1|
|convolution2d_3 (Convolution2D)|	(None, 5, 37, 48)	|43248	|convolution2d_2|
|convolution2d_4 (Convolution2D)|	(None, 3, 35, 64)	|27712	|convolution2d_3|
|dropout_1 (Dropout) |(None, 3, 35, 64)|0|convolution2d_4|
|convolution2d_5 (Convolution2D)|	(None, 1, 33, 64)	|36928	|dropout_1|
|dropout_2 (Dropout) |	(None, 1, 33, 64)	|0	|convolution2d_5|
|flatten_1 (Flatten)|	(None, 2112)|	0	|dropout_2|
|dense_1 (Dense)|	(None, 50)	|105650	|flatten_1|
|dropout_3 (Dropout) |	(None, 50)	|0	|dense_1|
|dense_2 (Dense)|	(None, 5)	|255	|dropout_3|
|dropout_4 (Dropout) |	(None, 5)	|0	|dense_2|
|dense_3 (Dense)|	(None, 1)	|6	|dropout_4|


Total params: 237,259
sTrainable params: 237,259
Non-trainable params: 0


#### 2. Attempts to reduce overfitting in the model
In order to reduce overfitting, I used data augmentation.
Multi-view camera is used, and mirrored image is created as below:

###### Center Camera
![alt text][image1]

###### Left Camera
![alt text][image2]

###### Right Camera
![alt text][image3]

In these side camera information, steering angle is modified so that the vehicle tries to keep the center of the road.
The variable `correction` is added/subtracted from original training data.

###### Mirroring
This circuit data mostly contains counter clockwise driving, so I need to augment data by inversing the image and training value.

###### Code Example
Multi viewing and mirroring was done as follows:

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

Value of the variable `correction` is 0.20. Because when I use 0.15, the vehicle fail to take strong curve before the bridge.


#### 3. Model parameter tuning

The model used an adam optimizer, so the learning rate was not tuned manually.

#### 4. Appropriate training data
Basically, I used pre-captured dataset provided by Udacity.
But even after eliminating overfitting, the vehicle couldn't pass through the bridge and dirt area.
So I decided to augment these two parts by copy/pasting so that the amount of such a rare situation gains more effect in the final model.


***

## Model Architecture and Training Strategy

#### 1. Solution Design Approach
Firstly, I used nvidia model first, and it turned out that there are overfitting (multi-viewing and mirroing was already done before this 1st attempt)).
Then I did network architecture modification by reducing the complexity of network, with dropout.

Then I could elimiate overfitting.

But, when I tried the model, the vehicle couldn't pass the bridge, or course out to dirt after the bridge. This problem couldn't be solved easily by architecture modification.
So I tried to augment such a rare case by simply copying and pasting. (more precisely, I modified `driving_log.csv`)

In the final model, you can see the train/test error has similar trend via iteration.

![alt text][image4]


#### 2. Output Movie

![alt text][video1]

Here's a [link to my video result](./output_video.mp4). 