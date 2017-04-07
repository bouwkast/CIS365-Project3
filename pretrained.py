from keras.applications.inception_v3 import InceptionV3
from keras.callbacks import ModelCheckpoint, EarlyStopping
from keras.preprocessing import image
from keras.models import Model
from keras.layers import Dense, GlobalAveragePooling2D
from keras import backend as K

# create the base pre-trained model
from keras.preprocessing.image import ImageDataGenerator

# TODO - clean up comments and make more that are relevant

base_model = InceptionV3(weights='imagenet', include_top=False)
batch_size = 16
dim = 299  # InceptionV3 is trained on 299x299 images

# add a global spatial average pooling layer
x = base_model.output
x = GlobalAveragePooling2D()(x)
# let's add a fully-connected layer
x = Dense(1024, activation='relu')(x)

# TODO - pick whichever is correct for the number of classes
# predictions = Dense(102, activation='softmax')(x)
predictions = Dense(17, activation='softmax')(x)

# this is the model we will train
model = Model(input=base_model.input, output=predictions)

# first: train only the top layers (which were randomly initialized)
# i.e. freeze all convolutional InceptionV3 layers
for layer in base_model.layers:
    layer.trainable = False

# compile the model (should be done *after* setting layers to non-trainable)
model.compile(optimizer='rmsprop', loss='categorical_crossentropy', metrics=['accuracy'])

train_datagen = ImageDataGenerator(
        rescale=1./255,
        zoom_range=[.65, 1],
        horizontal_flip=True,
        fill_mode='constant')

# this is the augmentation configuration we will use for testing:
# only rescaling
#
test_datagen = ImageDataGenerator(rescale=1./255)
# train the model on the new data for a few epochs
train_generator = train_datagen.flow_from_directory(
        '17_flowers/train',  # this is the target directory
        target_size=(dim, dim),  # all images will be resized to 150x150
        batch_size=batch_size,
        class_mode='categorical')  # since we use binary_crossentropy loss, we need binary labels

# this is a similar generator, for validation data
validation_generator = test_datagen.flow_from_directory(
        '17_flowers/validation',
        target_size=(dim, dim),
        batch_size=batch_size,
        class_mode='categorical')
print(validation_generator.class_indices)

# TODO - maybe mess around with how many epochs here
model.fit_generator(
        train_generator,
        steps_per_epoch=2000 // batch_size,
        epochs=2,
        validation_data=validation_generator,
        validation_steps=800 // batch_size)

# at this point, the top layers are well trained and we can start fine-tuning
# convolutional layers from inception V3. We will freeze the bottom N layers
# and train the remaining top layers.

# let's visualize layer names and layer indices to see how many layers
# we should freeze:
for i, layer in enumerate(base_model.layers):
   print(i, layer.name)

# TODO - maybe mess around with which layers we freeze/train
# we chose to train the top 2 inception blocks, i.e. we will freeze
# the first 172 layers and unfreeze the rest:
for layer in model.layers[:172]:
   layer.trainable = False
for layer in model.layers[172:]:
   layer.trainable = True

# we need to recompile the model for these modifications to take effect
# we use SGD with a low learning rate
# TODO - could try some other optimizer - I like rmsprop - can also change the parameters (not loss or metric)
from keras.optimizers import SGD
model.compile(optimizer=SGD(lr=0.0001, momentum=0.9), loss='categorical_crossentropy', metrics=['accuracy'])

# we train our model again (this time fine-tuning the top 2 inception blocks
# alongside the top Dense layers
# this is the augmentation configuration we will use for testing:
# only rescaling
test_datagen = ImageDataGenerator(rescale=1./255)

# this is a generator that will read pictures found in
# subfolers of 'data/train', and indefinitely generate
# batches of augmented image data

# TODO - change epochs and patience to your leisure
# TODO - can also mess with # of steps and the overal batch_size
model.fit_generator(
        train_generator,
        steps_per_epoch=2000 // batch_size,
        epochs=50,
        validation_data=validation_generator,
        validation_steps=800 // batch_size,
        callbacks=[EarlyStopping(patience=50), ModelCheckpoint('model.h5', verbose=1, save_best_only=True)])

# print(model.summary())

# this is a comment