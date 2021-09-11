#Resnet

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


#creating a residula network with 3 CNN blocks
class ResBlock(layers.Layer):
  def __init__(self, channels):
    super(ResBlock, self).__init__()
    self.cnn1 = CNNBlock(channels[0])
    self.cnn2 = CNNBlock(channels[1])
    self.cnn3 = CNNBlock(channels[2])
    self.pooling = layers.MaxPooling2D()
    self.identity_mapping = layers.Conv2D(channels[1], 1, padding = 'same')

  def call(self, input_tensor, training = False):
    x = self.cnn1(input_tensor, training = training)
    x = self.cnn2(x, training = training)
    x = self.cnn3(
        x + self.identity_mapping(input_tensor), training = training,
    )
    return self.pooling(x)

class Resnet_Like(keras.Model):
  #keras.Model has added functionalities like built in training, evaluation added to the func.s of layers.Layer
  #in the final model, keras.Model is to be used
  def __init__(self, num_classes = 10):
    super(Resnet_Like, self).__init__()
    #specifying the channels for each of the CNN blocks
    self.block1 = ResBlock([32,32,64])
    self.block2 = ResBlock([128,128,256])
    self.block3 = ResBlock([128,256,512])
    self.pool = layers.GlobalAveragePooling2D()
    self.classifier = layers.Dense(num_classes)

  def call(self, input_tensor, training = False):
    x = self.block1(input_tensor, training = training)
    x = self.block2(x, training = training)
    x = self.block3(x, training = training)
    x = self.pool(x)
    return self.classifier(x)
  #function is to provide shape to the outputs in the model summary
  def model(self):
    x = keras.Input(shape = (28,28,1))
    return keras.Model(inputs = [x], outputs = self.call(x))


model = Resnet_Like(num_classes = 10)
model.compile(
    optimizer = keras.optimizers.Adam(),
    loss = keras.losses.SparseCategoricalCrossentropy(from_logits=True),
    metrics = ["accuracy"],
)

print(model.model().summary())

model.fit(x_train, y_train, batch_size = 64, epochs = 1, verbose = 2)

model.evaluate(x_test, y_test, batch_size = 64, verbose = 2)
