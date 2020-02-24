###Jared Homer, Alex Stephens

from keras import models
from keras import layers
from keras import optimizers
import matplotlib.pyplot as plt
import numpy as np

red_dots = np.array([[1, 5], [2, 4], [7, 7], [4, 6], [6, 4]], "float32")  #Manual point entries
blue_dots = np.array([[6, 9], [4, 2], [8, 6], [5, 5], [3, 8]], "float32")

input = np.concatenate((red_dots, blue_dots), axis=0)  #Create vector with the first half red points and second half blue
normal_input = np.copy(input)
colors = np.array([[0, 1]])
count = 1
for i in range(0, np.size(red_dots[:, 0])-1):  #Create color vector with one entry for every point
    colors = np.concatenate((colors, [[0, 1]]), axis=0)   #Red dot = [0 1]
for i in range(0, np.size(blue_dots[:, 0])):
    colors = np.concatenate((colors, [[1, 0]]), axis=0)   #Blue dot = [1 0]

for i in range(0, 2):
   normal_input[:, i] = (input[:, i] - np.min(input[:,i])) / (np.max(input[:,i]) - np.min(input[:, i]))  #Normalizing input data X and Y separatly
#input[:, :] = (input[:, :] - np.min(input[:, :])) / (np.max(input[:, :]) - np.min(input[:, :]))

network = models.Sequential()  #Set up network with 3 layers, 2 inputs, 80 hidden layer nodes, and a softmax output.
network.add(layers.Dense(80, activation='sigmoid', input_shape=(2,)))
network.add(layers.Dense(2, activation='softmax'))  #Could also have been single output with sigmoid
optimizers.RMSprop(lr=0.00001)  #Defined learning rate for rmsprop optimizer
network.compile(loss='categorical_crossentropy',
              optimizer='rmsprop',
              metrics=['accuracy'])

history = network.fit(normal_input, colors, epochs=17000)  #20000
test_loss, test_acc = network.evaluate(normal_input, colors)
print('Test Accuracy:', test_acc)

acc = history.history['acc']
loss = history.history['loss']
epochs = range(1, len(acc) + 1)

step_size = 0.1
step_interval = int(10 / step_size)
size = step_interval * step_interval
step_divider = int(step_interval / 10)

predict_input = np.ndarray(shape=(size, 2))  #Setting up vector of all points for prediction output
for i in range(0, int(step_interval)):
    for j in range(0, int(step_interval)):
        predict_input[j + i * step_interval, 0] = i/step_divider
        predict_input[j + i * step_interval, 1] = j/step_divider
for i in range(0,2):
    predict_input[:, i] = (predict_input[:, i] - np.min(input[:, i])) / (np.max(input[:, i]) - np.min(input[:, i]))
#predict_input[:, :] = (predict_input[:, :] - np.min(predict_input[:, :])) / (np.max(predict_input[:, :]) - np.min(predict_input[:, :]))

plt.figure(1)
classes = network.predict(predict_input)
for i in range(0, step_interval):
    for j in range(0, step_interval):
        if classes[j + (i * step_interval), 0] > classes[j + (i * step_interval), 1]:  #Prints blue dot if class ~ [1 0]
            plt.plot(i/step_divider, j/step_divider, 'bo', markersize=1)
        else:
            plt.plot(i/step_divider, j/step_divider, 'ro', markersize=1)  #Otherwise prints red dot
    if i == 0:
        progress = 0
    else:
        progress = (i / step_interval) * 100

    print("Plot Progress: %.0f" % progress, "%")

plt.scatter(red_dots[:, 0], red_dots[:, 1], c="red")
plt.scatter(blue_dots[:, 0], blue_dots[:, 1], c="blue")
axes = plt.gca()
axes.set_ylabel('y')
axes.set_xlabel('x')
axes.set_title('Classification Region')
axes.set_xlim([0, 10])
axes.set_ylim([0, 10])
plt.show()

plt.figure(2)
plt.plot(epochs, acc, 'r', label='Training accuracy')
plt.title('Training accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()
plt.show()

plt.figure(3)
plt.plot(epochs, loss, 'b', label='Training loss')
plt.title('Training loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.show()








