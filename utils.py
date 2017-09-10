import csv
import cv2
import numpy as np

lines = []

with open('data/driving_log.csv') as csvfile:
	reader = csv.reader(csvfile)
	for line in reader:
		lines.append(line)

# print(lines[0])

images = []
measurements = []
i=0
for line in lines:
	source_path = line[0]
	# if i<5:
	# 	print(line)
	# 	print(source_path)
	# 	i+=1
	# filename = source_path.split('/')[-1]
	# current_path = 'data/IMG' + filename
	image = cv2.imread(source_path)
	images.append(image)
	measurement = float(line[3])
	measurements.append(measurement)


augmented_images, augmented_meaurements = [], []
for image, measurement in zip(images,measurements):
	augmented_images.append(image)
	augmented_meaurements.append(measurement)
	augmented_images.append(cv2.flip(image,1))
	augmented_meaurements.append(measurement*-1.0)

X_train = np.array(augmented_images)
y_train = np.array(augmented_meaurements)

# print(X_train.shape)
##Temporarily declaring model here for testing. To be included in models.py

from keras.models import Sequential
from keras.layers import Flatten, Dense, Lambda, Activation
from keras.layers.convolutional import Conv2D
from keras.layers.pooling import MaxPooling2D

model = Sequential()
model.add(Lambda(lambda x:x / 255.0 - 0.5, input_shape = (160,320,3)))
model.add(Conv2D(6, 5, 5, input_shape=(160, 320, 3)))
model.add(MaxPooling2D(pool_size=(2,2)))
model.add(Activation('relu'))
model.add(Flatten())
model.add(Dense(1))


model.compile(loss = 'mse', optimizer = 'adam')
model.fit(X_train, y_train, validation_split=0.2, shuffle=True, nb_epoch=7)

model.save('model.h5')