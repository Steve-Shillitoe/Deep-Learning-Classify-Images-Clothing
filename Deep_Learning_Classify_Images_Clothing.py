import tensorflow as tf

# Import TensorFlow Datasets
import tensorflow_datasets as tfds
tfds.disable_progress_bar()

# Helper libraries
import math
import numpy as np
import matplotlib.pyplot as plt

import logging
logger = tf.get_logger()
logger.setLevel(logging.ERROR)

def normalize(images, labels):
    #Normalises pixel values in the range [0,1]
    images = tf.cast(images, tf.float32)
    images /= 255
    return images, labels

#Import the Fashion MNIST dataset
dataset, metadata = tfds.load('fashion_mnist', as_supervised=True, with_info=True)
train_dataset, test_dataset = dataset['train'], dataset['test']

class_names = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat',
 'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']

num_train_examples = metadata.splits['train'].num_examples
num_test_examples = metadata.splits['test'].num_examples
print("Number of training examples: {}".format(num_train_examples))
print("Number of test examples: {}".format(num_test_examples))

# The map function applies the normalize function to each element in the train
# and test datasets
train_dataset = train_dataset.map(normalize)
test_dataset = test_dataset.map(normalize)

# The first time you use the dataset, the images will be loaded from disk
# Caching will keep them in memory, making training faster
train_dataset = train_dataset.cache()
test_dataset = test_dataset.cache()

# Take a single image, and remove the color dimension by reshaping
for image, label in test_dataset.take(1):
 break
image = image.numpy().reshape((28,28))
# Plot the image - voila a piece of fashion clothing
plt.figure()
plt.imshow(image, cmap=plt.cm.binary)
plt.colorbar()
plt.grid(False)
plt.show()

plt.figure(figsize=(10,10))
i = 0
for (image, label) in test_dataset.take(25):
     image = image.numpy().reshape((28,28))
     plt.subplot(5,5,i+1)
     plt.xticks([])
     plt.yticks([])
     plt.grid(False)
     plt.imshow(image, cmap=plt.cm.binary)
     plt.xlabel(class_names[label])
     i += 1
plt.show()

model = tf.keras.Sequential([
 # Input layer
    tf.keras.layers.Flatten(input_shape=(28, 28, 1)),
    tf.keras.layers.Dense(128, activation=tf.nn.relu),
    tf.keras.layers.Dense(10, activation='softmax')
 ])

#Compile the model
model.compile(optimizer='adam',
 loss=tf.keras.losses.SparseCategoricalCrossentropy(),
 metrics=['accuracy'])

#Train the model
BATCH_SIZE = 32
train_dataset = train_dataset.cache().shuffle(num_train_examples).batch(BATCH_SIZE)
test_dataset = test_dataset.cache().batch(BATCH_SIZE)

model.fit(train_dataset, epochs=5, validation_data=test_dataset)

#Evaluate accuracy
test_loss, test_accuracy = model.evaluate(test_dataset, steps=math.ceil(num_test_examples/32))
print('Accuracy on test dataset:', test_accuracy)

#Make prediction
for test_images, test_labels in test_dataset.take(1):
     test_images = test_images.numpy()
     test_labels = test_labels.numpy()
     predictions = model.predict(test_images)

print('The confidence of the model that the image corresponds to each of the 10 different articles of clothing = {}'.format(predictions[0]))
print('The most likely class = {}'.format(class_names[np.argmax(predictions[0])]))