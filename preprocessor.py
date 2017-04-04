"""

Filename: preprocessor.py
Author: Steven
Date Last Modified: 4/4/2017
Email: bouwkast@mail.gvsu.edu

"""

# TODO mess with some more values
from keras.preprocessing.image import ImageDataGenerator, array_to_img, img_to_array, load_img

datagen = ImageDataGenerator(
        rescale=1/255,
        # rotation_range=40,
        width_shift_range=0.2,
        # height_shift_range=0.2,
        # shear_range=1,
        zoom_range=[.65, 1],
        horizontal_flip=True,
        fill_mode='constant')

# CHANGE DIRECTORY AND IMAGE FOR TESTING BEST METHODS
img = load_img('17_flowers/train/bluebell/image_0251.jpg')  # this is a PIL image
x = img_to_array(img)
x = x.reshape((1,) + x.shape)  # this is a Numpy array with shape (1, 3, 150, 150)

# the .flow() command below generates batches of randomly transformed images
# and saves the results to the `preview/` directory
i = 0

# CHANGE THE PREFIX WHEN CHANGING IMAGE
for batch in datagen.flow(x, batch_size=1,
                          save_to_dir='preview', save_prefix='bluebell_test', save_format='jpeg'):
    i += 1
    if i > 20:
        break  # otherwise the generator would loop indefinitely