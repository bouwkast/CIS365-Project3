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


def create_percentages(probabilities):
    """
    Take a numpy array containing the probabilities of some other input
    data for what appropriate flower class it belongs to.
    :param probabilities: a numpy array of float values which are probabilities
    :return: a numpy array of float values as percentages
    """
    sum = np.sum(probabilities)
    percentages = []  # standard python list to contain the percentages

    # to calculate the percentage take each independent probability and divide it by the sum of all
    for prob in np.nditer(probabilities):
        percentages.append((prob / sum) * 100)

    return percentages


model = load_model('model.h5')
dim = 299

predict_datagen = ImageDataGenerator(rescale=1./255)

predict_generator = predict_datagen.flow_from_directory(
        'predict',
        target_size=(dim, dim),
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

percentages = create_percentages(prediction)
print(predicted_flower, percentages[best])

# prediction = model.predict(validation_generator, batch_size=1, verbose=1)

print(percentages)

print()
