#Convolutional Networks

import tensorflow as tf
import math
from tensorflow import keras
from tensorflow.keras import layers, regularizers
from tensorflow.keras.datasets import cifar10
#cifar10 is a dataset consisting of 10 classes of objects with 50k training images & 10k test images with 32*32 RGB
e = math.e

(x_train, y_train), (x_test, y_test) = cifar10.load_data()
x_train = x_train.astype("float32") / 255.0 
#this is being done to normalise the values of greyscale b/w 0 and 1, and float64 is being converted to float32 for ease in computation
x_test = x_test.astype("float32") / 255.0

#since it is convolutional so we're not going to flatten it
#sequential api approach
# model = keras.Sequential(
#     [
#       keras.Input(shape=(32,32,3)),
#       layers.Conv2D(32,3, padding = 'valid', activation = 'relu'),
#       layers.MaxPooling2D(pool_size = (2,2)),
#       layers.Conv2D(64,3,activation = 'relu'),
#       layers.MaxPooling2D(),
#       layers.Conv2D(128, 3, activation = 'relu'),
#       layers.Flatten(),
#       layers.Dense(64,activation = 'relu'),
#       layers.Dense(10),
#     ]
# )

#functional api, with batch normalisation
def my_model():
  #l2 is a batch normalisation method as well
  inputs = keras.Input(shape=(32,32,3))
  x = layers.Conv2D(
      32,3, padding = 'same', kernel_regularizer = regularizers.l2(0.01),            
                    )(inputs)
  x = layers.BatchNormalization()(x)
  x = keras.activations.relu(x)
  x = layers.MaxPooling2D()(x)
  x = layers.Conv2D(
      64, 3,padding = 'same', kernel_regularizer = regularizers.l2(0.01),
      )(x)
  x = layers.BatchNormalization()(x)
  x = keras.activations.relu(x)
  x = layers.Conv2D(
      128,3,padding = 'same', kernel_regularizer = regularizers.l2(0.01),
      )(x)
  x = layers.BatchNormalization()(x)
  x = keras.activations.relu(x)
  x = layers.Flatten()(x)
  x = layers.Dense(
      64 , activation='relu', kernel_regularizer = regularizers.l2(0.01),
      )(x)
  x = layers.Dropout(0.5)(x) #drops .5 part of the connections in b/w those layers
  outputs = layers.Dense(10)(x)
  model = keras.Model(inputs = inputs , outputs = outputs)
  return model

model = my_model()
#print(model.summary())
#when overfitting, we regularise the model by using dropout, early stoppage etc
# for that we import regularizers from keras
model.compile(
    loss = keras.losses.SparseCategoricalCrossentropy(from_logits= True),
    optimizer = keras.optimizers.Adam(lr = 3*e - 4),
    metrics = ["accuracy"],
)

model.fit(x_train, y_train,batch_size=64,epochs = 10, verbose = 2)
model.evaluate(x_test, y_test, batch_size= 64, verbose = 2)
