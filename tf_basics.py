import tensorflow as tf

#a tensor is an n-dim array of data
# 1d : vector
# 2d : matrix
# etc.

#initialisation of tensors
x = tf.constant(4)
# print(x) => tf.Tensor(4, shape=(), dtype=int32)...this has no shape
#we can also specify the shape

y = tf.constant(4, shape = (1,1))
#print(y) => tf.Tensor([[4]], shape=(1, 1), dtype=int32)

x = tf.ones((3,3))
# print(x)

# tf.Tensor(
# [[1. 1. 1.]
#  [1. 1. 1.]
#  [1. 1. 1.]], shape=(3, 3), dtype=float32)..same will be case for tf.zeros

# tf.eye(n) makes an identity matrix of dimension 1

x = tf.random.normal((3,3), mean  = 0, stddev= 1)
#above is for normal dist of values across tensors

x = tf.random.uniform((1,3),minval = 0, maxval = 1)
#gives tensor with random values from 0 to 1 

x = tf.range(9) # => tf.Tensor([0 1 2 3 4 5 6 7 8], shape=(9,), dtype=int32)

x = tf.range(start= 1, limit = 10,delta = 2) 
#print(x) => #tf.Tensor([1 3 5 7 9], shape=(5,), dtype=int32)

#although type can be specified while initialisation, it can be casted too
x = tf.cast(x, dtype = tf.float64)
#print(x) -> tf.Tensor([1. 3. 5. 7. 9.], shape=(5,), dtype=float64)
#tf.float(16,32,64), tf.int(8,16,32,64), tf.bool

#mathematical operations
x = tf.constant([1,2,3])
y = tf.constant([9,8,7])

z = tf.add(x,y) #-> tf.Tensor([10 10 10], shape=(3,), dtype=int32)
#z = x+y also works
#for subtraction, z = x-y or tf.subtract(x,y)

#for division, elementwise division : tf.divide(x,y)
#or z = x/y

#for multiplication, z = tf.multiply(x,y)
#or z = x*y

z = tf.tensordot(x, y, axes = 1) #->tf.Tensor(46, shape=(), dtype=int32)
#print(z)#this will do element wise multiplication and them add them
#other way of achieving the same is:
# z = tf.reduce_sum(x*y, axis = 0)
# print(z) -> tf.Tensor(46, shape=(), dtype=int32)

z = x ** 5 #element wise exponantiation
#print(z) -> tf.Tensor([  1  32 243], shape=(3,), dtype=int32)
 
x = tf.random.normal((2,3))
y = tf.random.normal((3,4))
z = tf.matmul(x,y) #matrix multiplication, it can also be done by using  z = x@y

#indexing
x = tf.constant([0,1,2,3,4,5])
# print(x[:]) #this will print all the elements
# print(x[1:]) #normal slicing rules, everything except first element
# print(x[1:3])#prints from 1 to 3
# #suppose we wanna skip one element alternatively
# print(x[::2])
print(x[::-1]) #reverse order

#how to print a sub set of the array with desrired values
indices = tf.constant([0,2])
x_ind = tf.gather(x, indices)
print(x_ind)

x = tf.constant([[1,2],
                [3,4]])

print(x[0,1:]) #0th row and all columns except first

#reshaping
x = tf.range(9)

x = tf.reshape(x , (3,3)) #reshaping
print(x)

x = tf.transpose(x, perm=[1,0])
print(x)
