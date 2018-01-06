
# coding: utf-8

# In[1]:

import numpy as np
import os
import csv
import cv2
import sklearn
from sklearn.model_selection import train_test_split
import datetime
import random


# In[2]:

# new 3/31/2017
# Read the data

csv_filepath = '/home/carnd/CarND-Behavioral-Cloning/data/driving_log.csv'
# did not need
# csv_filepath_1 = '/home/carnd/CarND-Behavioral-Cloning/data/driving_log_10.csv'

samples = []

with open(csv_filepath) as csvfile:
    next(csvfile)
    reader = csv.reader(csvfile)
    for line in reader:
        samples.append(line)
        
def add_to_samples(csv_filepath, samples):
    with open(csv_filepath) as csvfile:
        reader = csv.reader(csvfile)
        for line in reader:
            samples.append(line)
    return samples


# did not need as model worked without adding more data
# samples = add_to_samples(csv_filepath_1, samples)

print("Samples: ", len(samples))    

print("Completed:" , datetime.datetime.now())



# In[3]:

# new 3/31/2017
# get Center, Left and Right Images
# new added 03262017

import matplotlib.pyplot as plt
import matplotlib.image as mpimg


center =[]
left_turn = []
right_turn = []


for line in samples:
      
    keep_prob = random.random()
    # Normal right turns  - Double by adding small random fluctuations
    if (float(line[3]) >0.20 and float(line[3]) <=0.50):

        for j in range(2):
            new_steering =float(line[3])*(1.0 + np.random.uniform(-1,1)/100.0)
            right_turn.append([line[0], line[1], line[2], new_steering])

    # Normal left turns  -  Double by adding small random fluctuations

    elif (float(line[3]) >= -0.50 and float(line[3]) < -0.15):

        for j in range(2):
            new_steering = float(line[3])*(1.0 + np.random.uniform(-1,1)/100.0)
            left_turn.append([line[0], line[1], line[2], new_steering])

    ## Zero angle steering  - undersample by 10% 
    elif (float(line[3]) > -0.02 and float(line[3]) < 0.02):
        if keep_prob <=0.90:
            center.append([line[0], line[1], line[2], line[3]])

    else:
        center.append([line[0], line[1], line[2], line[3]])
        

new_list = []
new_list = right_turn + left_turn + center
print("Lengths:", len(new_list), len(center), len(left_turn), len(right_turn))

print("Completed:" , datetime.datetime.now())


# In[4]:

# new 3/31/2017
# get images


import matplotlib.pyplot as plt
import matplotlib.image as mpimg

images = []
angles = []

print("Starting image processing...")

for one_line in new_list:
    # center image
    center_source_path = one_line[0]
    center_filename = center_source_path.split('/')[-1]
    center_current_path = '/home/carnd/CarND-Behavioral-Cloning/data/IMG/' + center_filename
    center_image = cv2.imread(center_current_path)
       
    # Left Image
    left_source_path = one_line[1]
    if len(left_source_path) != 0:
        left_filename = left_source_path.split('/')[-1]
        left_current_path = '/home/carnd/CarND-Behavioral-Cloning/data/IMG/' + left_filename
        left_image = cv2.imread(left_current_path)

    # Right Image
    right_source_path = one_line[2]
    if len(right_source_path) != 0:
        right_filename = right_source_path.split('/')[-1]
        right_current_path = '/home/carnd/CarND-Behavioral-Cloning/data/IMG/' + right_filename
        right_image = cv2.imread(right_current_path)
    
    center_angle = float(one_line[3])
    
    # Apply correction for left and right steering
    correction = 0.20
    left_angle = center_angle + correction
    right_angle = center_angle - correction
    
    # Randomly include either center, left or right image
    num = random.random()
    if len(left_source_path) == 0 and len(right_source_path) == 0:
        select_image = center_image
        select_angle = center_angle
        images.append(select_image)
        angles.append(select_angle)
    else:
        if num <= 0.33:
            select_image = center_image
            select_angle = center_angle
            images.append(select_image)
            angles.append(select_angle)
        elif num>0.33 and num<=0.66:
            select_image = left_image
            select_angle = left_angle
            images.append(select_image)
            angles.append(select_angle)
        else:
            select_image = right_image
            select_angle = right_angle
            images.append(select_image)
            angles.append(select_angle)
        
    # Randomly horizontally flip selected images with 80% probability
    keep_prob = random.random()
    if keep_prob >0.20:
        flip_image = np.fliplr(select_image)
        flip_angle = -1*select_angle
        images.append(flip_image)
        angles.append(flip_angle)

        
print("Images: ", len(images)) 
print("Angles: ", len(angles)) 
    
x_train = np.array(images)
y_train = np.array(angles)

print("X train: ", len(x_train)) 
print("Y train: ", len(y_train)) 


print("Completed:" , datetime.datetime.now())


# In[5]:

# new 3/31/2017
# Model #2 

from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten, SpatialDropout2D, ELU, Lambda
from keras.layers.convolutional import Convolution2D
from keras.regularizers import l2
from keras.layers.core import Activation, Reshape
from keras.layers import Convolution2D, MaxPooling2D, Cropping2D
from keras.layers import core, convolutional, pooling
from keras import models, optimizers, backend

print("Starting model processing....")

model = Sequential()
model.add(Lambda(lambda x: x/255.0 - 0.5, input_shape=(160, 320, 3)))
model.add(Cropping2D(cropping=((70, 25),(0, 0))))

model.add(convolutional.Convolution2D(16, 3, 3, input_shape=(32, 128, 3), activation='relu'))
model.add(pooling.MaxPooling2D(pool_size=(2, 2)))
model.add(convolutional.Convolution2D(32, 3, 3, activation='relu'))
model.add(pooling.MaxPooling2D(pool_size=(2, 2)))
model.add(convolutional.Convolution2D(64, 3, 3, activation='relu'))
model.add(pooling.MaxPooling2D(pool_size=(2, 2)))
model.add(core.Flatten())
model.add(core.Dense(500, activation='relu'))
model.add(core.Dropout(.5))
model.add(core.Dense(100, activation='relu'))
model.add(core.Dropout(.25))
model.add(core.Dense(20, activation='relu'))
model.add(core.Dense(1))
model.compile(optimizer=optimizers.Adam(lr=1e-04), loss='mean_squared_error')


model.compile(loss= 'mse', optimizer = 'adam')
model.summary()

model.fit(x_train, y_train, validation_split = 0.2, shuffle=True, nb_epoch=2)

model.save('model.h5')


print("Saved model to disk")

