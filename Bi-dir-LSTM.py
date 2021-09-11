import tensorflow as tf
from tensorflow import keras
import math
from tensorflow.keras import layers
from tensorflow.keras.datasets import mnist

e = math.e
(x_train, y_train), (x_test, y_test) = mnist.load_data()

x_train = x_train.astype("float32") / 255.0 #this is being done to normalise the values of greyscale b/w 0 and 1, and float64 is being converted to float32 for ease in computation
x_test = x_test.astype("float32") / 255.0

#in the first timestamp it'll send the first row and in 2nd, second row shall be sent etc.
model = keras.Sequential()
model.add(keras.Input(shape = (None, 28))) #28 timestamps, none is here because we don't need a specific timestamp...28 pixels for each timestamp
model.add(
    layers.Bidirectional(
      layers.LSTM(256, return_sequences = True, activation = 'tanh') #multiple RNN layers on top of each other shall be stacked 
    )     #since, it is bidriectional, we'll get 512 nodes instead of 256,
    #one time it will be forward, other time it will be backwards hence doubled
 ) #default activation is tanh
model.add(
    layers.Bidirectional(
      layers.LSTM(256, activation = 'tanh')
    )
)
model.add(layers.Dense(10))

print(model.summary())

model.compile(
    loss = keras.losses.SparseCategoricalCrossentropy(from_logits= True),
    optimizer = keras.optimizers.Adam(lr = 0.001),
    metrics = ["accuracy"],
)

model.fit(x_train, y_train,batch_size=64,epochs = 10, verbose = 2)
model.evaluate(x_test, y_test, batch_size= 64, verbose = 2)
