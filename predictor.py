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

model = load_model('model_70percent.h5')
model.summary()


predict_datagen = ImageDataGenerator(rescale=1./255)

predict_generator = predict_datagen.flow_from_directory(
        'predict',
        target_size=(150, 150),
        batch_size=1,
        class_mode=None)


prediction = model.predict_generator(predict_generator, 1)

names = {'buttercup': 1, 'tigerlily': 14, 'bluebell': 0, 'crocus': 4, 'daisy': 6, 'snowdrop': 12, 'lily_valley': 10, 'tulip': 15, 'daffodil': 5, 'iris': 9, 'pansy': 11, 'colts_foot': 2, 'fritillary': 8, 'dandelion': 7, 'cowslip': 3, 'windflower': 16, 'sunflower': 13}

#prediction should be numpy array of probabilites, find highest, record index

best = np.argmax(prediction)
print(best)
predicted_flower = 'ERROR'
for key in names:
    if names[key] == best:
        predicted_flower = key

print(predicted_flower)

# prediction = model.predict(validation_generator, batch_size=1, verbose=1)

print(prediction)