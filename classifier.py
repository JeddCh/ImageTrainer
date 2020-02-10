from keras_preprocessing.image import ImageDataGenerator
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D
from keras.layers import Activation, Dropout, Flatten, Dense
from keras import backend as K
import numpy as np
from keras.preprocessing import image

width = 200
height = 200
train_folder = "pictures/classes"
valid_folder = "pictures/validation"
samples = 633
validation_samples = 100
epochs = 80
batch = 5

if K.image_data_format() == 'channels_first':
    shape = (3, width, height)
else:
    shape = (width, height, 3)

train_Data = ImageDataGenerator(
    rescale=1./255,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True)

test_Data = ImageDataGenerator(rescale=1./255)

train_generator = train_Data.flow_from_directory(
    train_folder,
    target_size=(width, height),
    batch_size=batch,
    class_mode='binary')

valid_generator = test_Data.flow_from_directory(
    valid_folder,
    target_size=(width, height),
    batch_size=batch,
    class_mode='binary')

model = Sequential()
model.add(Conv2D(32, (3, 3), input_shape=shape))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))

model.summary()

model.add(Conv2D(32, (3, 3), input_shape=shape))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Conv2D(64, (3, 3), input_shape=shape))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Flatten())
model.add(Dense(64))
model.add(Activation('relu'))
model.add(Dropout(0.5))
model.add(Dense(1))
model.add(Activation('sigmoid'))

model.summary()

model.compile(loss='binary_crossentropy',
              optimizer='rmsprop',
              metrics=['accuracy'])

model.fit_generator(
    train_generator,
    steps_per_epoch=samples//batch,
    epochs=epochs,
    validation_data=valid_generator,
    validation_steps=validation_samples//batch)

model.save_weights('hot_dog.h5')

prediction = image.load_img('pictures/validation/hotdog/1200px-Chili_dog_with_fries.jpg', target_size=(200, 200))
prediction = image.img_to_array(prediction)
prediction = np.expand_dims(prediction, axis=0)

result = model.predict(prediction)
print(result)
if result[0][0] == 1:
    final = "hotdog"
else:
    final = "not hotdog"

print(final)




