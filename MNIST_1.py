#building basic neural nets

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.datasets import mnist

(x_train, y_train), (x_test, y_test) = mnist.load_data()
#print(x_train.shape)
#now we'd need to flatten them to have only one long column for those feature values
#so we reshape it to same num of rows and change it to

x_train = x_train.reshape(-1, 28*28).astype("float32") / 255.0 #this is being done to normalise the values of greyscale b/w 0 and 1, and float64 is being converted to float32 for ease in computation
x_test = x_test.reshape(-1, 28*28).astype("float32") / 255.0

#Sequential API of keras(very convenient but not very flexible)
model = keras.Sequential(
    [
      #keras.Input(shape = (28*28)), #this sends a demo input to be able to let model print its summary
     layers.Dense(512, activation = 'relu'),
     layers.Dense(256, activation = 'relu'),
     layers.Dense(10),
    ]
)

# print(model.summary()) #model.summary is a common debugging tool
# import sys
# sys.exit()

# another way of adding layers is :
model = keras.Sequential()
model.add(keras.Input(shape=(784)))
model.add(layers.Dense(512, activation = 'relu'))
model.add(layers.Dense(256, activation = 'relu'))
model.add(layers.Dense(10))

#extracting specific layer outputs:
# model = keras.Model(inputs = model.inputs,
#                     outputs = [model.layers[-1].output]) #index value decides which layer to take, -1 means last, -2 means second last and henceforth
#if there are names of the layers by name argument, then they can be accessed in te=he following way:
#[model.get_layer('name of the layer').output]

#for all layers, outputs = [layer.output for layer in model.layers]
#features = model.predict(x_train)
#for feature in features:
#   print(feature.shape)


# feature = model.predict(x_train)
# print(feature.shape)
# import sys
# sys.exit()

#Functional API (A bit more flexible)
inputs = keras.Input(shape = (784))
x = layers.Dense(512, activation = 'relu')(inputs)
x = layers.Dense(256, activation = 'relu')(x)
outputs = layers.Dense(10, activation = 'softmax')(x)
model = keras.Model(inputs = inputs, outputs = outputs)

#compile is about the specifications of the model
model.compile(
    #Use this crossentropy loss function when there are two or more label classes. 
    #We expect labels to be provided as integers.
    #from_logits is true to send the last layer through a softmax activation first
    loss = keras.losses.SparseCategoricalCrossentropy(from_logits= False), 
    optimizer = keras.optimizers.Adam(lr = 0.001),
    metrics = ["accuracy"],
)

#fit is more about training and preparing it
model.fit(x_train, y_train, batch_size = 32, epochs = 5, verbose = 2)
model.evaluate(x_test, y_test, batch_size = 32, verbose = 2)
