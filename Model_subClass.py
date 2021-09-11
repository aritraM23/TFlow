#Model Subclassing

#increases flexibility

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.datasets import mnist

(x_train, y_train), (x_test, y_test) = mnist.load_data()
#print(x_train.shape)
#now we'd need to flatten them to have only one long column for those feature values
#so we reshape it to same num of rows and change it to

x_train = x_train.reshape(-1, 28,28,1).astype("float32") / 255.0 #this is being done to normalise the values of greyscale b/w 0 and 1, and float64 is being converted to float32 for ease in computation
x_test = x_test.reshape(-1, 28,28,1).astype("float32") / 255.0

# CNN-> BatchNornm -> ReLU (common structure)
# what if we write the structure in class because running timestamp wise would take high computational power

#subclasing is the same way of using pytorch

class CNNBlock(layers.Layer):
  def __init__(self, out_channels, kernel_size = 3):
    super(CNNBlock, self).__init__() #runs parent class layer by layer
    self.conv = layers.Conv2D(out_channels, kernel_size, padding = 'same') #Conv layer
    self.bn = layers.BatchNormalization() #BatchNorm layer

  def call(self, input_tensor, training = False):
    #call method is the forward method, which takes the input tensor and runs the layers
    x = self.conv(input_tensor)
    x = self.bn(x, training = training)
    x = tf.nn.relu(x)
    return x


model = keras.Sequential(
    [
      CNNBlock(32),
      CNNBlock(64),
      CNNBlock(128),
      layers.Flatten(),
      layers.Dense(10),
    ]
)

model.compile(
    optimizer = keras.optimizers.Adam(),
    loss = keras.losses.SparseCategoricalCrossentropy(from_logits=True),
    metrics = ["accuracy"],
)

model.fit(x_train, y_train, batch_size = 64, epochs = 3, verbose = 2)
model.evaluate(x_test, y_test, batch_size = 64, verbose = 2)
