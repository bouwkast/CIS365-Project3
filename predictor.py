"""

Filename: predictor.py
Author: Steven
Date Last Modified: 4/3/2017
Email: bouwkast@mail.gvsu.edu

"""
from keras.models import load_model
from keras.preprocessing.image import ImageDataGenerator, array_to_img, img_to_array, load_img
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D
from keras.layers import Activation, Dropout, Flatten, Dense
from keras.callbacks import TensorBoard, ModelCheckpoint
import numpy as np

model = load_model('model.h5')
model.summary()

img = load_img('preview/bluebell_test_0_389.jpeg')  # this is a PIL image
x = np.array(img)
# x = img_to_array(img)
print(x.shape)
x = x.reshape((1,) + x.shape)  # this is a Numpy array with shape (1, 3, 150, 150)






prediction = model.predict(x, batch_size=1, verbose=1)

print(prediction)