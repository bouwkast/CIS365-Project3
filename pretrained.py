from keras.applications.inception_v3 import InceptionV3
from keras.callbacks import ModelCheckpoint, EarlyStopping
from keras.preprocessing import image
from keras.models import Model
from keras.layers import Dense, GlobalAveragePooling2D
from keras import backend as K

# Create the base pre-trained model
from keras.preprocessing.image import ImageDataGenerator

base_model = InceptionV3(weights='imagenet', include_top=False)
batch_size = 32
dim = 299  # InceptionV3 is trained on 299x299 images

# Add a global spatial average pooling layer
x = base_model.output
x = GlobalAveragePooling2D()(x)
# Add a fully-connected layer
x = Dense(1024, activation='relu')(x)

# Select 102 or 17 for whichever model you want to train on.
predictions = Dense(102, activation='softmax')(x)
# predictions = Dense(17, activation='softmax')(x)

# This is the model we will train
model = Model(input=base_model.input, output=predictions)

# First: train only the top layers (which were randomly initialized)
# i.e. freeze all convolutional InceptionV3 layers
for layer in base_model.layers:
    layer.trainable = False

# Compile the model (should be done *after* setting layers to non-trainable)
model.compile(optimizer='rmsprop', loss='categorical_crossentropy', metrics=['accuracy'])

# Formats the pictures so they will be read the same way.
train_datagen = ImageDataGenerator(
        rescale=1./255,
        horizontal_flip=True,
        fill_mode='constant')

# This is the augmentation configuration we will use for testing:
# only rescaling
test_datagen = ImageDataGenerator(rescale=1./255)
# Train the model on the new data for a few epochs
train_generator = train_datagen.flow_from_directory(
        '102_flowers/train',  # this is the target directory
        target_size=(dim, dim),  # all images will be resized to 299x299
        batch_size=batch_size,
        class_mode='categorical')  # since we use binary_crossentropy loss, we need binary labels

# This is a similar generator, for validation data
validation_generator = test_datagen.flow_from_directory(
        '102_flowers/validation',
        target_size=(dim, dim),
        batch_size=batch_size,
        class_mode='categorical')
print(validation_generator.class_indices)

# Trains all user added layers before actually reading the pictures
model.fit_generator(
        train_generator,
        steps_per_epoch=2000 // batch_size,
        epochs=2,
        validation_data=validation_generator,
        validation_steps=800 // batch_size)

# At this point, the top layers are well trained and we can start fine-tuning
# convolutional layers from inception V3. We will freeze the bottom N layers
# and train the remaining top layers.

# Let's visualize layer names and layer indices to see how many layers we should freeze:
for i, layer in enumerate(base_model.layers):
   print(i, layer.name)

# We chose to train the top 2 inception blocks, i.e. we will freeze
# the first 172 layers and unfreeze the rest. This prevents overfitting.
for layer in model.layers[:172]:
   layer.trainable = False
for layer in model.layers[172:]:
   layer.trainable = True

# We need to recompile the model for these modifications to take effect
# We use SGD with a low learning rate
from keras.optimizers import SGD
model.compile(optimizer=SGD(lr=0.0001, momentum=0.9), loss='categorical_crossentropy', metrics=['accuracy'])

# We train our model again (this time fine-tuning the top 2 inception blocks
# alongside the top Dense layers
# This is the augmentation configuration we will use for testing:
# only rescaling
test_datagen = ImageDataGenerator(rescale=1./255)

# This is a generator that will read pictures found in
# subfolers of 'data/train', and indefinitely generate
# batches of augmented image data
model.fit_generator(
        train_generator,
        steps_per_epoch=2000 // batch_size,
        epochs=200,
        validation_data=validation_generator,
        validation_steps=800 // batch_size,
        callbacks=[EarlyStopping(patience=50), ModelCheckpoint('102_model3.h5', verbose=1, save_best_only=True)])