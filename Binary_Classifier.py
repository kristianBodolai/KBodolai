'''based on Sentdex's tutorial, optimized for high accuracy (around 94% for dogs vs cats binary classification)'''

import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras import Model
from tensorflow.keras.layers import Dense, Dropout, Activation, Flatten, Conv2D, MaxPooling2D
import pickle
from tensorflow.keras.callbacks import TensorBoard

'''Because my GPU has low memory, it only supports 65x65 images, training is 7 times faster, but accuracy decreases around 2% because of lower image resolution. This code works as it is, if it's run on a more powerful GPU, go to loading_data.py and up the IMG_SIZE'''

NAME = 'A7'
print("GPU is", "available" if tf.config.experimental.list_physical_devices("GPU") else "NOT AVAILABLE")
#
tensorboard = TensorBoard(log_dir = "logs/{}".format(NAME))

X = pickle.load(open("X.pickle", "rb"))
y = pickle.load(open("y.pickle", "rb"))

X = X/255.0

model = Sequential()

model.add(Conv2D(filters=30, kernel_size=3, input_shape = X.shape[1:], activation = 'relu'))
model.add(MaxPooling2D(pool_size = (4,4)))
model.add(Dropout(.25))

model.add(Conv2D(60, 3, activation = 'relu'))
model.add(MaxPooling2D(pool_size = (2,2)))
model.add(Dropout(.25))

model.add(Conv2D(120, 3, activation = 'relu'))
model.add(MaxPooling2D(pool_size = (2,2)))
model.add(Dropout(.25))

model.add(Flatten())

model.add(Dense(300, activation='relu'))
model.add(Dropout(.25))

model.add(Dense(150, activation='relu'))
model.add(Dropout(.25))


model.add(Dense(1, activation = "sigmoid"))

model.compile(loss = 'binary_crossentropy', optimizer = 'adam', metrics = ['accuracy'])

model.fit(X, y, batch_size = 32, epochs = 200, validation_split = .2, callbacks = [tensorboard])
#0.93 with C30C60C120D1200 0.2 dropout 1st conv 4x4 pool_size, rest, 2x2

model.summary()
