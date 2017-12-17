import os
import csv
import cv2
import numpy as np
import sklearn
from keras.models import Sequential
from keras.layers import Flatten, Dense, Dropout, ELU
from keras.layers.core import Activation
from keras.layers import Lambda, MaxPooling2D
from keras.layers.convolutional import Convolution2D
from keras.layers import Cropping2D
from keras.layers.normalization import BatchNormalization

from keras.models import Model
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from keras.optimizers import Adam

samples = []

"""
with open('./data/driving_log.csv', 'r') as f:
    reader = csv.reader(f)
    for row in reader:
        steering_center = float(row[3])

        # create adjusted steering measurements for the side camera images
        correction = 0.2 # this is a parameter to tune
        steering_left = steering_center + correction
        steering_right = steering_center - correction

        # read in images from center, left and right cameras
        path = "..." # fill in the path to your training IMG directory
        img_center = process_image(np.asarray(Image.open(path + row[0])))
        img_left = process_image(np.asarray(Image.open(path + row[1])))
        img_right = process_image(np.asarray(Image.open(path + row[2])))

        # add images and angles to data set
        car_images.extend(img_center, img_left, img_right)
        steering_angles.extend(steering_center, steering_left, steering_right)
"""

with open('./data/driving_log.csv') as csvfile:
    reader = csv.reader(csvfile)
    header = next(reader)  # skip first line

    for line in reader:
        samples.append(line)

from sklearn.model_selection import train_test_split
train_samples, validation_samples = train_test_split(samples, test_size=0.2)

def data_augment(samples):

    return

def generator(samples, batch_size=32):
    num_samples = len(samples)
    while 1: # Loop forever so the generator never terminates
        sklearn.utils.shuffle(samples)
        for offset in range(0, num_samples, batch_size):
            batch_samples = samples[offset:offset+batch_size]

            images = []
            angles = []
            correction = 0.20
            for batch_sample in batch_samples:
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

                images.append(center_image)
                angles.append(center_angle)
                images.append(left_image)
                angles.append(left_angle)
                images.append(right_image)
                angles.append(right_angle)
                images.append(inverse_center_image)
                angles.append(inverse_center_angle)
                images.append(inverse_left_image)
                angles.append(inverse_left_angle)
                images.append(inverse_right_image)
                angles.append(inverse_right_angle)

            # trim image to only see section with road
            X_train = np.array(images)
            y_train = np.array(angles)
            yield sklearn.utils.shuffle(X_train, y_train)

# compile and train the model using the generator function
train_generator = generator(train_samples, batch_size=128)
validation_generator = generator(validation_samples, batch_size=128)


model = Sequential()


model.add(Lambda(lambda x:x/255.0 - 0.5, input_shape=(160, 320, 3)))
model.add(Cropping2D(cropping = ((70,25), (0, 0))))
model.add(Convolution2D(24, 5, 5, subsample=(2,2), activation="relu"))
model.add(Convolution2D(36, 5, 5, subsample=(2,2), activation="relu"))
model.add(Convolution2D(48, 5, 5, subsample=(2,2), activation="relu"))
model.add(Convolution2D(64, 3, 3, activation="relu"))
model.add(Dropout(0.5))
model.add(Convolution2D(64, 3, 3, activation="relu"))
model.add(Dropout(0.5))
model.add(Flatten())
model.add(Dense(50))
model.add(Dropout(0.5))
model.add(Dense(5))
model.add(Dropout(0.5))
model.add(Dense(1))

model.compile(loss='mse', optimizer='adam')

print("summary\n", model.summary())
#model.fit_generator(train_generator, samples_per_epoch=len(train_samples)*6,
# validation_data=validation_generator, nb_val_samples=len(validation_samples), nb_epoch=10)

#model.save('model.h5')

#
# Data Visualization
#


history_object = model.fit_generator(train_generator, samples_per_epoch =len(train_samples)*6, 
    validation_data = validation_generator, nb_val_samples = len(validation_samples), 
    nb_epoch=10, verbose=1)
model.save('model.h5')

### print the keys contained in the history object
print(history_object.history.keys())

### plot the training and validation loss for each epoch
plt.plot(history_object.history['loss'])
plt.plot(history_object.history['val_loss'])
plt.title('model mean squared error loss')
plt.ylabel('mean squared error loss')
plt.xlabel('epoch')
plt.legend(['training set', 'validation set'], loc='upper right')
plt.savefig("./output.png")
