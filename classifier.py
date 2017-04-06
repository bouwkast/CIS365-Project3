# #  From - https://blog.keras.io/building-powerful-image-classification-models-using-very-little-data.html
#
from keras.preprocessing.image import ImageDataGenerator, array_to_img, img_to_array, load_img



import datetime
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D
from keras.layers import Activation, Dropout, Flatten, Dense
from keras.callbacks import TensorBoard, ModelCheckpoint
from keras.utils import np_utils
now = datetime.datetime.now()
tensorboard = TensorBoard(log_dir='./logs/' + now.strftime('%Y.%m.%d %H.%M'))
#  alternate architecture
dim = 224
# conv = Sequential()
# conv.add(Conv2D())

# add our layers
model = Sequential()
model.add(Conv2D(32, (3, 3), border_mode='valid', input_shape=(150, 150, 3)))

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
print(validation_generator.class_indices)

model.fit_generator(
        train_generator,
        steps_per_epoch=2000 // batch_size,
        epochs=50,
        validation_data=validation_generator,
        validation_steps=800 // batch_size,
        callbacks=[ModelCheckpoint('model.h5', verbose=1, save_best_only=True)])

model.summary()  # print out summary of model for testing purposes

# TODO - remove this save_weights, ModelCheckpoint will save the actual model
# model.save_weights('first_try.h5')  # always save your weights after training or during training

#this is a comment