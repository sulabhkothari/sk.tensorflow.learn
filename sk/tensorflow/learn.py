from __future__ import absolute_import, division, print_function, unicode_literals
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf

mammal = tf.Variable("Elephant", tf.string)
print(mammal)

mymat = tf.Variable([[7],[11]], tf.int16)
print(mymat)

myxor = tf.Variable([[False, True],[True, False]], tf.bool)
print(myxor)

linear_squares = tf.Variable([[4], [9], [16], [25]], tf.int32)
print(linear_squares)

squarish_squares = tf.Variable([ [4, 9], [16, 25] ], tf.int32)
print(squarish_squares)

rank_of_squares = tf.rank(squarish_squares)
print(rank_of_squares)

mymatC = tf.Variable([[7],[11]], tf.int32)
print(mymatC)

print("Rank of the last Tensor -> {}, Shape -> {}", tf.rank(mymatC), mymatC.shape)

print(mymatC.shape[1])

from tensorflow import keras
from tensorflow.keras import layers

import tensorflow_datasets as tfds
tfds.disable_progress_bar()

embedding_layer = layers.Embedding(1000, 5)

result = embedding_layer(tf.constant([1,2,3]))
result.numpy()


import matplotlib.pyplot as plt
import numpy as np
import tensorflow.compat.v2 as tf

import tensorflow_datasets as tfds

#from tensorflow.python.framework.ops import disable_eager_execution

tfds.disable_progress_bar()

tf.enable_v2_behavior()

print(tfds.list_builders())

mnist_train = tfds.load(name="mnist", split="train")
assert isinstance(mnist_train, tf.data.Dataset)
print(mnist_train)

mnist = tfds.load("mnist:1.*.*")
for mnist_example in mnist_train.take(1):  # Only take a single example
  image, label = mnist_example["image"], mnist_example["label"]

  plt.imshow(image.numpy()[:, :, 0].astype(np.float32), cmap=plt.get_cmap("gray"))
  print("Label: %d" % label.numpy())

ds, ds_info = tfds.load('cifar10', split='train', with_info=True)
#fig = tfds.show_examples(ds_info, ds)

# Strip leading and trailing 2 elements
foo = tf.constant([1,2,3,4,5,6])
print(foo[2:-2])  # => [3,4]

# Skip every other row and reverse the order of the columns
foo = tf.constant([[1,2,3], [4,5,6], [7,8,9]])
print(foo[::2,::-1])  # => [[3,2,1], [9,8,7]]

# Use scalar tensors as indices on both dimensions
print(foo[tf.constant(0), tf.constant(2)])  # => 3

# Insert another dimension
foo = tf.constant([[1,2,3], [4,5,6], [7,8,9]])
print(foo[tf.newaxis, :, :]) # => [[[1,2,3], [4,5,6], [7,8,9]]]
print(foo[:, tf.newaxis, :]) # => [[[1,2,3]], [[4,5,6]], [[7,8,9]]]
print(foo[:, :, tf.newaxis]) # => [[[1],[2],[3]], [[4],[5],[6]],


# Ellipses (3 equivalent operations)
foo = tf.constant([[1,2,3], [4,5,6], [7,8,9]])
print(foo[tf.newaxis, :, :])  # => [[[1,2,3], [4,5,6], [7,8,9]]]
print(foo[tf.newaxis, ...])  # => [[[1,2,3], [4,5,6], [7,8,9]]]
print(foo[tf.newaxis])  # => [[[1,2,3], [4,5,6], [7,8,9]]]

# Masks
foo = tf.constant([[1,2,3], [4,5,6], [7,8,9]])
print(foo[foo > 2])  # => [3, 4, 5, 6, 7, 8, 9]



rank_three_tensor = tf.ones([3, 4, 5])
matrix = tf.reshape(rank_three_tensor, [6, 10])  # Reshape existing content into
                                                 # a 6x10 matrix
matrixB = tf.reshape(matrix, [3, -1])  #  Reshape existing content into a 3x20
                                       # matrix. -1 tells reshape to calculate
                                       # the size of this dimension.
matrixAlt = tf.reshape(matrixB, [4, 3, -1])  # Reshape existing content into a
                                             #4x3x5 tensor

# Note that the number of elements of the reshaped Tensors has to match the
# original number of elements. Therefore, the following example generates an
# error because no possible value for the last dimension will match the number
# of elements.

print(matrixAlt.dtype)

# Cast a constant integer tensor into floating point.
float_tensor = tf.cast(tf.constant([1, 2, 3]), dtype=tf.float32)
print(float_tensor)

#yet_another = tf.reshape(matrixAlt, [13, 2, -1])  # ERROR!


constant = tf.constant([1, 2, 3])
tensor = constant * constant
print(tensor)
