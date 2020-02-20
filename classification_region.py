###Jared Homer, Alex Stephens

from keras import models
from keras import layers
import matplotlib.pyplot as plt
import numpy as np

red_dots = np.array([[1, 5], [2, 4], [7, 7], [4, 6], [6, 4]])
blue_dots = np.array([[6, 9], [4, 2], [8, 6], [5, 5], [3, 8]])

plt.scatter(red_dots[:,0], red_dots[:,1],c="red")
plt.scatter(blue_dots[:,0], blue_dots[:,1],c="blue")
axes = plt.gca()
axes.set_ylabel('y')
axes.set_xlabel('x')
axes.set_title('Classification Region')
axes.set_xlim([0, 10])
axes.set_ylim([0, 10])
plt.show()

#network = models.Sequential()
#network.add(layers.Dense(100, activation='relu', input_shape=(10 * 10,)))
#network.add(layers.Dense(2, activation='softmax'))




