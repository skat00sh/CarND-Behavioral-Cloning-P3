import csv
import cv2
import numpy as np
import matplotlib.pyplot as plt


# images = []
# angles = []
measurements = []
samples = []
testimg = []
#open and read train data, which is provided by official

with open('/home/devendra/udacity/sdc-nd-t1/CarND-Behavioral-Cloning-P3/data/driving_log.csv') as csvfile:
    reader = csv.reader(csvfile)
    for line in reader:
        samples.append(line)


#flip images so that get more data and prevent our car to aways go aside
def AugmentImages(images,measurements):
    augmented_images,augmented_measurements = [],[]
    # count = 0
    for image,measurement in zip(images,measurements):
        augmented_images.append(image)
        augmented_measurements.append(measurement)
        augmented_images.append(cv2.flip(image,1))
        augmented_measurements.append(measurement*-1.0)
        # count += 1
        # print(count)
    X_train = np.array(augmented_images)
    y_train = np.array(augmented_measurements)
    # plt.imshow(images[0])
    # plt.show()
    # plt.imshow(augmented_images[1])
    # plt.show()
    return X_train,y_train


#split train data and valid data
from sklearn.model_selection import train_test_split
train_samples, validation_samples = train_test_split(samples, test_size=0.2)


import sklearn.utils
#use generator so that we needn't load all data in memory, which need huge memory source
def generator(samples, batch_size=32):
    num_samples = len(samples)
    while 1: 
        sklearn.utils.shuffle(samples)
        for offset in range(0, num_samples, batch_size):
            batch_samples = samples[offset:offset+batch_size]
            images = []
            angles = []
            for batch_sample in batch_samples:
                #use images and angles from three cameras, so that we can get more data to train
                center_name = '/home/devendra/udacity/sdc-nd-t1/CarND-Behavioral-Cloning-P3/data/IMG/'+batch_sample[0].split('/')[-1]
                center_image = cv2.imread(center_name)
                center_angle = float(batch_sample[3])
                images.append(center_image)
                angles.append(center_angle)

                correction = 0.1 # this is a parameter to tune

                left_name = '/home/devendra/udacity/sdc-nd-t1/CarND-Behavioral-Cloning-P3/data/IMG/'+batch_sample[1].split('/')[-1]
                left_image = cv2.imread(left_name)
                left_angle = center_angle + correction
                images.append(left_image)
                angles.append(left_angle)

                right_name = '/home/devendra/udacity/sdc-nd-t1/CarND-Behavioral-Cloning-P3/data/IMG/'+batch_sample[2].split('/')[-1]
                right_image = cv2.imread(right_name)
                right_angle = center_angle - correction
                images.append(right_image)
                angles.append(right_angle)
                
                X_train = np.array(images)
                y_train = np.array(angles)
                X_train,y_train = AugmentImages(X_train,y_train)
            
            yield sklearn.utils.shuffle(X_train, y_train)

        # trim image to only see section with road
        


train_generator = generator(train_samples, batch_size=32)
# print(train_generator.
validation_generator = generator(validation_samples, batch_size=32)
X_batch, y_batch = next(train_generator)


#import keras modules
from keras.models import Sequential
from keras.layers import Flatten,Dense
from keras.layers import Lambda
from keras.layers import Cropping2D
from keras.layers import Conv2D
from keras.layers import Dropout


input_shape = (X_batch.shape[1], X_batch.shape[2], X_batch.shape[3])
pool_size = (2,2)
#model architecture
model = Sequential()
# model.add(Cropping2D(cropping=((50,20), (0,0)),input_shape=(160,320,3)))

# model.add(Lambda(lambda x:(x/255.0 - 0.5)))
model.add(Lambda(lambda x: x/127.5 - 1.,
            input_shape=input_shape,
output_shape=input_shape))
model.add(Conv2D(24,(5,5),strides = (2, 2),activation = 'relu'))
model.add(Conv2D(36,(5,5),strides = (2, 2),activation = 'relu'))
model.add(Conv2D(48,(5,5),strides = (2, 2),activation = 'relu'))
model.add(Conv2D(64,(3,3),activation = 'relu')) #had stride of (1,1)
model.add(Conv2D(64,(3,3),activation = 'relu')) #had stride of (1,1)
model.add(Flatten())
# model.add(Dense(1164,activation = 'relu'))
# model.add(Dropout(0.3))
model.add(Dense(100,activation = 'relu'))
model.add(Dropout(0.3))
model.add(Dense(50,activation = 'relu'))
model.add(Dropout(0.3))
model.add(Dense(10,activation = 'relu'))
model.add(Dropout(0.3))
model.add(Dense(1))

#model training and validing
model.compile(loss='mse',optimizer='adam')
# samples_per_epoch= len(train_samples*3*2)  ===> train_samples*Number of camera (Here we have left right and center) * Flipping every image so doubling the sample
history_object = model.fit_generator(train_generator, 
                                        samples_per_epoch= len(train_samples*3*2), 
                                        validation_data=validation_generator, 
                                        nb_val_samples=len(validation_samples), 
                                        nb_epoch=10,verbose=1)


#visualize training process
print(history_object.history.keys())
plt.plot(history_object.history['loss'])
plt.plot(history_object.history['val_loss'])
plt.title('model mean squared error loss')
plt.ylabel('mean squared error loss')
plt.xlabel('epoch')
plt.legend(['training set', 'validation set'], loc='upper right')
plt.show()

#save model
model.save('model.h5')
