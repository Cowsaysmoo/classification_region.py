###Jared Homer, Alex Stephens
#######################################################################################
import numpy as np
import matplotlib.pyplot as plt
from keras import models
from keras import layers
from keras import utils
from keras import optimizers
#######################################################################################
# prepare samples
RED_SAMPLES = np.array([
    [1,5],
    [2,4],
    [7,7],
    [4,6],
    [6,4]
], dtype="float32")

BLUE_SAMPLES = np.array([
    [6,9],
    [4,2],
    [8,6],
    [5,5],
    [3,8]
], dtype="float32")

LABELS_SAMPLES = np.concatenate(
    (np.array(len(RED_SAMPLES)*[[0,1]], dtype="float32"), # Red output
    np.array(len(BLUE_SAMPLES)*[[1,0]], dtype="float32")), # Blue output
    axis=0
)
#######################################################################################

input_samples = np.concatenate((RED_SAMPLES, BLUE_SAMPLES), axis=0)

# normalize samples
norm_min_x = np.min(input_samples[:, 0])
norm_min_y = np.min(input_samples[:, 1])
diff_x_minmax = np.max(input_samples[:, 0]) - norm_min_x
diff_y_minmax = np.max(input_samples[:, 1]) - norm_min_y
normalized_samples = np.zeros(input_samples.shape)
for i, pt in enumerate(input_samples):
    normalized_samples[i, 0] = (pt[0] - norm_min_x) / diff_x_minmax
    normalized_samples[i, 1] = (pt[1] - norm_min_y) / diff_y_minmax

# Network architecture
net = models.Sequential()

# adding layers
net.add(layers.Dense(80, activation="sigmoid", input_shape=(2,))) # input will be a point through layer of 80 neurons
net.add(layers.Dense(2, activation="softmax")) # output will be 2 values, representing the confidence in color
net.compile(
    optimizer="rmsprop",
    loss="categorical_crossentropy",
    metrics=["acc"]
)

# save history and train network
net_history = net.fit(normalized_samples, LABELS_SAMPLES, epochs=25000)

# plot accuracy and loss plots
accuracy = net_history.history["acc"] # for some reason i need to type accuracy instead of acc
loss = net_history.history["loss"]
epochs = range(1, len(loss) + 1)

# test set
test_points_x, test_points_y = np.meshgrid(np.linspace(0,10,100), np.linspace(0,10,100))
test_points_flat_x = np.reshape(test_points_x, (test_points_x.size, 1))
test_points_flat_y = np.reshape(test_points_y, (test_points_y.size, 1))
test_points_flat = np.concatenate((test_points_flat_x, test_points_flat_y), axis=1)

# normalize test points
norm_test = np.zeros(test_points_flat.shape)
for i, pt in enumerate(test_points_flat):
    norm_test[i][0] = (pt[0] - norm_min_x) / diff_x_minmax
    norm_test[i][1] = (pt[1] - norm_min_y) / diff_y_minmax

# Predict class of given points
test_class = net.predict(norm_test)

# plot points according to their class
plt.figure(1)
plt.scatter(
    test_points_flat_x[test_class[:, 1] >= 0.5],
    test_points_flat_y[test_class[:, 1] >= 0.5],
    c='r', marker='.', s=0.25
)
plt.scatter(
    test_points_flat_x[test_class[:, 0] >= 0.5],
    test_points_flat_y[test_class[:, 0] >= 0.5],
    c='b', marker='.', s=0.25
)
plt.scatter(
    RED_SAMPLES[:, 0],
    RED_SAMPLES[:, 1],
    c='r', marker='o', s=1.5
)
plt.scatter(
    BLUE_SAMPLES[:, 0],
    BLUE_SAMPLES[:, 1],
    c='b', marker='o', s=1.5
)
plt.tight_layout()

# plot accuracy and loss of network from training
plt.figure(2)
plt.subplot(1,2,1)
plt.plot(epochs, accuracy, 'g', label="Training Accuracy")
plt.subplot(1,2,2)
plt.plot(epochs, loss, 'r', label="Training Loss")
plt.show()