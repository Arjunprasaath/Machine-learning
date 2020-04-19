
import keras
from keras.models import Sequential
from keras.layers import Convolution2D
from keras.layers import MaxPooling2D
from keras.layers import Flatten
from keras.layers import Dense

model = Sequential()

model.add(Convolution2D(filters=32,kernel_size=(2,2),data_format = 'channels_last',input_shape=(64,64,3) ,activation='relu'))

model.add( MaxPooling2D( pool_size=(2,2)))

model.add(Convolution2D(filters= 32,kernel_size=(2,2),activation='relu'))

model.add(MaxPooling2D(pool_size=(2,2)))

model.add(Flatten())

model.add(Dense(activation='relu',output_dim=128))

model.add(Dense(activation='relu',output_dim=128))

model.add(Dense(output_dim=1,activation='sigmoid'))

model.compile(optimizer='adam',loss='binary_crossentropy',metrics=['accuracy'])

from keras.preprocessing.image import ImageDataGenerator 
train_datagen = ImageDataGenerator(
        rescale=1./255,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True)

test_datagen = ImageDataGenerator(rescale=1./255)

training_set = train_datagen.flow_from_directory(
        'dataset/training_set',
        target_size=(64,64),
        batch_size=32,
        class_mode='binary')

test_set = test_datagen.flow_from_directory(
        'dataset/test_set',
        target_size=(64,64),
        batch_size=32,
        class_mode='binary')

model.fit_generator(
        training_set,
        steps_per_epoch=8000,
        epochs=20,
        validation_data= test_set,
        validation_steps=2000)
        
