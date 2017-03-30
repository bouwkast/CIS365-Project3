# #  From - https://blog.keras.io/building-powerful-image-classification-models-using-very-little-data.html
#
from keras.preprocessing.image import ImageDataGenerator, array_to_img, img_to_array, load_img
#
#
# # TODO mess with some more values
# datagen = ImageDataGenerator(
#         # rotation_range=40,
#         width_shift_range=0.2,
#         # height_shift_range=0.2,
#         # shear_range=1,
#         zoom_range=[.65, 1],
#         horizontal_flip=True,
#         fill_mode='constant')
#
# # CHANGE DIRECTORY AND IMAGE FOR TESTING BEST METHODS
# img = load_img('17_flowers/train/bluebell/image_0251.jpg')  # this is a PIL image
# x = img_to_array(img)  # this is a Numpy array with shape (3, 150, 150)
# x = x.reshape((1,) + x.shape)  # this is a Numpy array with shape (1, 3, 150, 150)
#
# # the .flow() command below generates batches of randomly transformed images
# # and saves the results to the `preview/` directory
# i = 0
#
# # CHANGE THE PREFIX WHEN CHANGING IMAGE
# for batch in datagen.flow(x, batch_size=1,
#                           save_to_dir='preview', save_prefix='bluebell_test', save_format='jpeg'):
#     i += 1
#     if i > 20:
#         break  # otherwise the generator would loop indefinitely
import datetime
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D
from keras.layers import Activation, Dropout, Flatten, Dense
from keras.callbacks import TensorBoard
from keras.utils import np_utils
now = datetime.datetime.now()
tensorboard = TensorBoard(log_dir='./logs/' + now.strftime('%Y.%m.%d %H.%M'))
# add our layers
model = Sequential()
model.add(Conv2D(32, (3, 3), input_shape=(150, 150, 3)))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Conv2D(32, (3, 3)))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Conv2D(64, (3, 3)))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Flatten())  # this converts our 3D feature maps to 1D feature vectors
model.add(Dense(64))
model.add(Activation('relu'))
model.add(Dropout(0.5))
model.add(Dense(17))
model.add(Activation('softmax'))

model.compile(loss='categorical_crossentropy',
              optimizer='rmsprop',
              metrics=['accuracy'])


batch_size = 16

# this is the augmentation configuration we will use for training
train_datagen = ImageDataGenerator(
        rescale=1./255,
        zoom_range=[.65, 1],
        horizontal_flip=True,
        fill_mode='constant')

# this is the augmentation configuration we will use for testing:
# only rescaling
test_datagen = ImageDataGenerator(rescale=1./255)

# this is a generator that will read pictures found in
# subfolers of 'data/train', and indefinitely generate
# batches of augmented image data
train_generator = train_datagen.flow_from_directory(
        '17_flowers/train',  # this is the target directory
        target_size=(150, 150),  # all images will be resized to 150x150
        batch_size=batch_size,
        class_mode='categorical')  # since we use binary_crossentropy loss, we need binary labels

# this is a similar generator, for validation data
validation_generator = test_datagen.flow_from_directory(
        '17_flowers/validation',
        target_size=(150, 150),
        batch_size=batch_size,
        class_mode='categorical')

model.fit_generator(
        train_generator,
        steps_per_epoch=2000 // batch_size,
        epochs=2,
        validation_data=validation_generator,
        verbose=1,
        validation_steps=800 // batch_size)
model.save_weights('first_try.h5')  # always save your weights after training or during training