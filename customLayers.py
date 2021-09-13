#building Custom layers

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.datasets import mnist

(x_train, y_train), (x_test, y_test) = mnist.load_data()
x_train = x_train.reshape(-1, 28*28).astype("float32") / 255.0 #this is being done to normalise the values of greyscale b/w 0 and 1, and float64 is being converted to float32 for ease in computation
x_test = x_test.reshape(-1, 28*28).astype("float32") / 255.0

#building custom models
class Dense(layers.Layer):
  def __init__(self, units ): #input_dim
    super(Dense, self).__init__()
    self.units = units
    
  #build dim, if given, then there would be no need of paramete input_Weight
  def build(self, input_shape):
    self.w = self.add_weight(
        name = 'w',
        shape = (input_shape[-1], self.units),
        initializer = 'random_normal',
        trainable = True,
    )

    self.b = self.add_weight(
        name = 'b',
        shape = (self.units),
        initializer = 'zeros',
        trainable = True,
    )

  def call(self,inputs):
    return tf.matmul(inputs, self.w) + self.b

#custom ReLu function
class MyReLu(layers.Layer):
  def __init__(self):
    super(MyReLu, self).__init__()

  def call(self, x):
    return tf.math.maximum(x,0)

class MyModel(keras.Model):
  def __init__(self, num_classes = 10):
    super(MyModel, self).__init__()
    # self.dense1 = layers.Dense(64)
    # self.dense2 = layers.Dense(num_classes)

    #custom layers with input dimension and without build method
    # self.dense1 = Dense(64,784)
    # self.dense2 = Dense(10,64)

    #with build method
    self.dense1 = Dense(64)
    self.dense2 = Dense(num_classes)
    self.relu = MyReLu()

  def call(self, x):
    x = self.relu(self.dense1(x))
    return self.dense2(x)

model = MyModel()
model.compile(
    optimizer = keras.optimizers.Adam(),
    loss = keras.losses.SparseCategoricalCrossentropy(from_logits=True),
    metrics = ["accuracy"],
)

model.fit(x_train, y_train, batch_size = 32, epochs = 2, verbose = 2)
model.evaluate(x_test, y_test, batch_size = 32, verbose = 2)
