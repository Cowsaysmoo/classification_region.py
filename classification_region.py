###Jared Homer, Alex Stephens

from keras import models
from keras import layers
from keras import optimizers
import matplotlib.pyplot as plt
import numpy as np

red_dots = np.array([[1, 5], [2, 4], [7, 7], [4, 6], [6, 4]], "float32")
blue_dots = np.array([[6, 9], [4, 2], [8, 6], [5, 5], [3, 8]], "float32")

input = np.concatenate((red_dots, blue_dots), axis=0)
normal_input = np.copy(input)
colors = np.array([[0, 1]])
count = 1
for i in range(0, np.size(red_dots[:, 0])-1):  #Create color vector with one entry for every point
    colors = np.concatenate((colors, [[0, 1]]), axis=0)
for i in range(0, np.size(blue_dots[:, 0])):
    colors = np.concatenate((colors, [[1, 0]]), axis=0)
for i in range(0, 2):
   normal_input[:, i] = (input[:, i] - np.min(input[:,i])) / (np.max(input[:,i]) - np.min(input[:, i]))  #Normalizing input data
#input[:, :] = (input[:, :] - np.min(input[:, :])) / (np.max(input[:, :]) - np.min(input[:, :]))
network = models.Sequential()
network.add(layers.Dense(80, activation='sigmoid', input_shape=(2,)))
network.add(layers.Dense(2, activation='softmax'))
optimizers.RMSprop(lr=0.00001)
network.compile(loss='categorical_crossentropy',
              optimizer='rmsprop',
              metrics=['accuracy'])

history = network.fit(normal_input, colors, epochs=40000)

predict_input = np.ndarray(shape=(2755, 2))
for i in range(0, 55):
    for j in range(0, 55):
        predict_input[j + i * 50, 0] = i/5
        predict_input[j + i * 50, 1] = j/5
for i in range(0,2):
    predict_input[:, i] = (predict_input[:, i] - np.min(input[:, i])) / (np.max(input[:, i]) - np.min(input[:, i]))
#predict_input[:, :] = (predict_input[:, :] - np.min(predict_input[:, :])) / (np.max(predict_input[:, :]) - np.min(predict_input[:, :]))
print(predict_input)
test_loss, test_acc = network.evaluate(normal_input, colors)
print('test_accuracy:', test_acc)

acc = history.history['acc']
loss = history.history['loss']
epochs = range(1, len(acc) + 1)

plt.figure(1)
classes = network.predict(predict_input)
for i in range(0, 55):
    for j in range(0, 55):
        if classes[j + (i * 50), 0] > classes[j + (i * 50), 1]:  #Blue
            plt.plot(i/5, j/5, 'bo', markersize=1)
        else:
            plt.plot(i/5, j/5, 'ro', markersize=1)

    print(i)

print(predict_input)
plt.scatter(red_dots[:, 0], red_dots[:, 1], c="red")
plt.scatter(blue_dots[:, 0], blue_dots[:, 1], c="blue")
axes = plt.gca()
axes.set_ylabel('y')
axes.set_xlabel('x')
axes.set_title('Classification Region')
#axes.set_xlim([0, 10])
#axes.set_ylim([0, 10])
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








