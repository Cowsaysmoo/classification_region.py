###Jared Homer, Alex Stephens

import numpy as np
import matplotlib.pyplot as plt
from keras import models
from keras import layers
from keras import utils
from keras import optimizers

# prepare samples
RED_SAMPLES = np.array([
    [1,5],
    [2,4],
    [7,7],
    [4,6],
    [6,4]
])
BLUE_SAMPLES = np.array([
    [6,9],
    [4,2],
    [8,6],
    [5,5],
    [3,8]
])
LABELS_SAMPLES = np.concatenate(
    (np.array(len(RED_SAMPLES)*[[0,1]]), # Red output
    np.array(len(BLUE_SAMPLES)*[[1,0]])), # Blue output
    axis=0
)

# Normalize samples
blue_normalized = utils.normalize(BLUE_SAMPLES, axis=-1, order=2)
red_normalized = utils.normalize(RED_SAMPLES, axis=-1, order=2)

normalized_samples = np.concatenate((red_normalized, blue_normalized), axis=0)

# Network architecture
net = models.Sequential()

# adding layers
net.add(layers.Dense(80, activation="relu", input_shape=(2,))) # input will be a point through layer of 80 neurons
net.add(layers.Dense(2, activation="softmax")) # output will be 2 values, representing the confidence in color
optimizers.RMSprop(lr=0.8)
net.compile(
    optimizer="rmsprop",
    loss="categorical_crossentropy",
    metrics=["accuracy"]
)

# save history and train network
net_history = net.fit(normalized_samples, LABELS_SAMPLES, epochs=20000)

# plot accuracy and loss plots
accuracy = net_history.history['accuracy'] # for some reason i need to type accuracy instead of acc
loss = net_history.history['loss']
epochs = range(1, len(loss) + 1)

plt.figure(1)
plt.subplot(1,2,1)
plt.plot(epochs, accuracy, 'g', label="Training Accuracy")
plt.subplot(1,2,2)
plt.plot(epochs, loss, 'r', label="Training Loss")
plt.show()