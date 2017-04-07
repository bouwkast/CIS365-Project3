"""

Filename: image_resizer.py
Author: Steven
Date Last Modified: 4/7/2017
Email: bouwkast@mail.gvsu.edu

"""
from keras.preprocessing.image import ImageDataGenerator, array_to_img, img_to_array, load_img
from scipy.misc import imresize


## works with pretrained model only
dim = 299

img = load_img('predict/to_predict/Daisy-2.jpg')

print(img_to_array(img).shape)

x = img_to_array(img)

x = imresize(x, (dim, dim, 3))  # resize it to the correct dimensions


#  TODO - need to reshape the image to have the following dimensions (None, dim, dim, 3)



x = x.reshape((1,) + x.shape)

# x = x.reshape((1, ) + (dim, dim, 3))

predict_d = ImageDataGenerator(rescale=1. / 255)

batch_size = 1

#  TODO - the dimensions get changed here too - is this where we can get (None, dim, dim, 3)???
for batch in predict_d.flow(x, batch_size=1):
    x = batch  # this runs in an infinite loop - need to stop it
    break

print(x.shape)